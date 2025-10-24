import torch
import torchvision.models as models
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoModel,
    AutoModelForSemanticSegmentation,
    CLIPTextModel 
)
import numpy as np
import pytest
import time
import os
from torch.utils import _pytree as pytree

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime

# 성능 측정 함수 
def measure_latency(func, warmup_runs: int, measure_runs: int):
    """
    func 함수를 받아서, warmup_runs 만큼 예열하고
    measure_runs 만큼 실제 실행 시간을 (ms) 단위로 측정합니다.
    """
    latencies = []
    
    # 1. Warmup
    for _ in range(warmup_runs):
        _ = func()
        
    # 2. 실제 측정
    for _ in range(measure_runs):
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
        
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    
    return avg_latency, max_latency, result


# 테스트 준비 함수 (conftest.py의 model_config를 주입받음)
@pytest.fixture(scope="module")
def setup_model_and_export(model_config):
    
    # 선택된 모델(CV 또는 LLM)을 동적으로 로드, Export, Compile, .pte 저장.
    model_name = model_config["name"]
    model_type = model_config["type"]
    model_id = model_config.get("model_id_for_load", model_name)
    pte_path = f"{model_name}.pte"
    print(f"\n[Setup] Loading model: {model_name} (Type: {model_type})")

    # 1. 모델 로드 및 입력 생성 (모델 타입에 따라 분기)
    if model_type == "cv":
        model = getattr(models, model_name)(weights="DEFAULT").eval()
        # CV 모델 입력
        example_args = (torch.randn(1, 3, 224, 224),)
        example_kwargs = {}
        pytorch_inputs = example_args 
        
    elif model_type == "llm":
        model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        token_inputs = tokenizer("Hello, world!", return_tensors="pt")
        # LLM 모델 입력
        example_args = (token_inputs['input_ids'],)
        example_kwargs = {'attention_mask': token_inputs['attention_mask']}
        pytorch_inputs = token_inputs # PyTorch 실행 시 **kwargs로 전달
    elif model_type == "llm_encoder":
        model = AutoModel.from_pretrained(model_id).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        token_inputs = tokenizer("Hello, world!", return_tensors="pt")

        example_args = (token_inputs['input_ids'],)
        example_kwargs = {'attention_mask': token_inputs['attention_mask']}
        pytorch_inputs = token_inputs
    elif model_type == "clip_text":
        # CLIP 모델의 '텍스트 부분'만 로드
        model = CLIPTextModel.from_pretrained(model_id).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        token_inputs = tokenizer("a photo of a cat", return_tensors="pt")
        
        example_args = (token_inputs['input_ids'],)
        example_kwargs = {'attention_mask': token_inputs['attention_mask']}
        pytorch_inputs = token_inputs
    elif model_type == "cv_segmentation":
        model = AutoModelForSemanticSegmentation.from_pretrained(model_id).eval()
        # SegFormer는 보통 512x512 또는 1024x1024 입력을 사용
        example_args = (torch.randn(1, 3, 512, 512),) 
        example_kwargs = {}
        pytorch_inputs = example_args # *args로 실행
    else:
        pytest.fail(f"Unsupported model type: {model_type}")

    # 2. Export (공통 로직)
    print(f"[Setup] Exporting {model_name}...")
    exported_program = torch.export.export(model, args=example_args, kwargs=example_kwargs)
    
    # 3. Compile/Lower (공통 로직)
    print(f"[Setup] Compiling {model_name} with XNNPACK...")
    program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    
    # 4. .pte 파일 저장 (공통 로직)
    with open(pte_path, "wb") as f:
        f.write(program.buffer)
    print(f"[Setup] {pte_path} 파일 저장 완료")
    
    # ExecuTorch 실행 시 사용할 입력 리스트
    et_inputs_list = list(example_args) + list(example_kwargs.values())
    
    # 테스트 함수에 필요한 모든 것을 yield
    yield model, pytorch_inputs, et_inputs_list, pte_path, model_config
    
    # 5. Teardown (테스트 종료 후 정리)
    if os.path.exists(pte_path):
        os.remove(pte_path)
        print(f"\n[Teardown] {pte_path} 파일 삭제 완료")


# 메인 테스트 함수
def test_model_verification(setup_model_and_export):
    
    model, pytorch_inputs, et_inputs_list, pte_path, config = setup_model_and_export
    
    # 설정값 로드
    model_name = config["name"]
    model_type = config["type"]
    measure_runs = config["measure_runs"]
    warmup_runs = config["warmup_runs"]
    assert_tolerance = config["assert_tolerance"]
    
    # 1. PyTorch 원본 모델 Latency 측정
    def pytorch_run():
        with torch.no_grad():
            if model_type == "cv" or model_type == "cv_segmentation":
                return model(*pytorch_inputs) # *args
            elif model_type == "llm" or model_type == "llm_encoder" or model_type == "clip_text":
                return model(**pytorch_inputs) # **kwargs

    pytorch_avg, pytorch_max, pytorch_result = measure_latency(pytorch_run, warmup_runs, measure_runs)
    
    
    # 2. ExecuTorch 모델 Latency 측정
    runtime = Runtime.get()
    method = runtime.load_program(pte_path).load_method("forward")
    
    def executorch_run():
        return method.execute(et_inputs_list)
    
    executorch_avg, executorch_max, executorch_result_list = measure_latency(executorch_run, warmup_runs, measure_runs)
    
    # 3. 결과 텐서 추출 (모델 타입에 따라 분기)
    executorch_result = executorch_result_list[0].clone()
    
    if model_type == "cv":
        pytorch_tensor = pytree.tree_flatten(pytorch_result)[0][0]
    elif model_type == "llm":
        pytorch_tensor = pytorch_result.logits
    elif model_type == "llm_encoder":
        pytorch_tensor = pytorch_result.last_hidden_state
    elif model_type == "clip_text":
        pytorch_tensor = pytorch_result.last_hidden_state
    elif model_type == "cv_segmentation":
        pytorch_tensor = pytorch_result.logits
    
    # 파일 핸들 명시적 해제 (PermissionError 방지)
    del method
    del runtime

    # 4. PyTorch/ExecuTorch 결과 비교 (공통 로직)
    pytorch_np = pytorch_tensor.cpu().numpy()
    executorch_np = executorch_result.cpu().numpy()
    
    mean_abs_diff = np.mean(np.abs(pytorch_np - executorch_np))
    
    print(f"\n--- Model Verification Results ({model_name}) ---")
    print(f"model_name: {model_name}")
    print(f"model_type: {model_type}")
    print(f"measure_runs: {measure_runs} (warmup: {warmup_runs})")
    print(f"assert_tolerance: < {assert_tolerance}")
    print(f"mean_absolute_difference: {mean_abs_diff:.6f}")
    print(f"pytorch_latency: avg {pytorch_avg:.2f} ms, max {pytorch_max:.2f} ms")
    print(f"executorch_latency: avg {executorch_avg:.2f} ms, max {executorch_max:.2f} ms")

    # 동적 assert 적용
    assert mean_abs_diff < assert_tolerance, f"Mean absolute difference ({mean_abs_diff}) is too high (Tolerance: {assert_tolerance})."
