import torch
import torchvision.models as models
import numpy as np
import pytest
import time
import os
from torch.utils import _pytree as pytree

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime

# 테스트 설정값 정의
MODEL_NAME = "resnet18"
PTE_PATH = f"{MODEL_NAME}.pte"
EXAMPLE_INPUTS = (torch.randn(1, 3, 224, 224),)

# 실제 성능을 측정하기 위해 50번 실행
MEASURE_RUNS = 50

# 성능 측정 함수 정의
def measure_latency(func):
    """
    func 함수(예: PyTorch 실행 함수)를 받아서,
    그 함수의 실행 시간을 (ms) 단위로 측정
    """
    latencies = []  # 측정한 시간 저장용 배열
    
    # CPU가 처음 일할 땐 느릴 수 있으니, 미리 10번 돌림 (결과값엔 포함 하지 않음)
    for _ in range(10):
        _ = func()
        
    # 실행 및 시간 측정
    for _ in range(MEASURE_RUNS):
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()
        
        # 초(s)를 밀리초(ms)로 변환하여 리스트에 추가
        latencies.append((end_time - start_time) * 1000)
        
    avg_latency = np.mean(latencies)  # 50번 측정한 시간의 평균
    max_latency = np.max(latencies)  # 50번 중 가장 오래 걸린 시간
    
    return avg_latency, max_latency, result


# 테스트 준비 함수(실행 시 최초 1회 실행)
@pytest.fixture(scope="module")
def setup_model_and_export():
    """
    1. ResNet18 모델 로드
    2. 모델을 ExecuTorch(.pte) 파일로 compile하고 저장
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    
    # Export: PyTorch 모델을 ExecuTorch가 이해할 수 있는 중간 단계로 변환
    exported_program = torch.export.export(model, EXAMPLE_INPUTS)
    
    # 중간 단계를 CPU(XNNPACK)에서 가장 빠르게 돌도록 최적화
    program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()]  # "XNNPACK CPU 최적화"
    ).to_executorch()
    
    # 최적화된 모델을 resnet18.pte 파일로 저장
    with open(PTE_PATH, "wb") as f:
        f.write(program.buffer)
    print(f"{PTE_PATH} 파일 저장 완료")
        
    # 테스트 함수에 전달
    yield model, EXAMPLE_INPUTS, PTE_PATH
    
    # 테스트 완료 후 정리
    if os.path.exists(PTE_PATH):
        os.remove(PTE_PATH)


# 메인 테스트 함수
def test_model_verification(setup_model_and_export):
    
    # setup_model_and_export에서 전달 받고 시작
    model, example_inputs, pte_path = setup_model_and_export
    
    # 1. PyTorch 원본 모델 Latency 측정
    def pytorch_run():  # Latency 측정에 넣어줄 추론 실행 함수 정의
        with torch.no_grad():  # no_grad는 추론모드
            return model(*example_inputs)  # pytorch 모델 객체에 example_inputs로 추론 실행

    pytorch_avg, pytorch_max, pytorch_result = measure_latency(pytorch_run)
    
    
    # 2. ExecuTorch 모델 Latency 측정
    runtime = Runtime.get()  # ExecuTorch 실행기 로드
    method = runtime.load_program(pte_path).load_method("forward") # 준비한 경로의 resnet18.pte 로드
    et_inputs = list(example_inputs) # ExecuTorch는 입력을 list 형태로 받기에 변환 
    
    def executorch_run():  # Latency 측정에 넣어줄 추론 실행 함수 정의
        return method.execute(et_inputs)  # ExecuTorch 실행
    
    executorch_avg, executorch_max, executorch_result_list = measure_latency(executorch_run)
    
    # 3. 결과값 저장
    # ExecuTorch는 결과를 리스트(list)로 반환, 리스트의 첫 번째 결과물 저장
    executorch_result = executorch_result_list[0].clone() 
    
    # ResNet18은 출력이 1개지만, 여러 개일 수 있으므로 pytree로 안전하게 꺼냅니다.
    pytorch_result_flat, _ = pytree.tree_flatten(pytorch_result)
    pytorch_tensor = pytorch_result_flat[0]  # PyTorch의 최종 결과 텐서


    # 4. PyTorch/ExecuTorch 결과 비교
    # Numpy 계산기로 계산하기 위해 두 텐서를 Numpy 배열로 변환
    pytorch_np = pytorch_tensor.cpu().numpy()
    executorch_np = executorch_result.cpu().numpy()
    
    # (PyTorch 결과 - ExecuTorch 결과)의 절대값(abs)을 구하고, 그 값들의 평균을 냄.
    mean_abs_diff = np.mean(np.abs(pytorch_np - executorch_np))
    
    print(f"\n--- Model Verification Results ---")
    print(f"model_name: {MODEL_NAME}")
    print(f"mean_absolute_difference: {mean_abs_diff:.6f}")
    print(f"pytorch_latency: avg {pytorch_avg:.2f} ms, max {pytorch_max:.2f} ms")
    print(f"executorch_latency: avg {executorch_avg:.2f} ms, max {executorch_max:.2f} ms")

    assert mean_abs_diff < 1e-5, f"Mean absolute difference ({mean_abs_diff}) is too high."