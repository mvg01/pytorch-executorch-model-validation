import pytest

# 모델별 테스트 설정을 중앙에서 관리.
MODEL_CONFIGS = {
    "resnet18": {
        "type": "cv",
        "measure_runs": 50,
        "warmup_runs": 10,
        "assert_tolerance": 1e-5, 
    },
    "mobilenet_v2": {
        "type": "cv",
        "measure_runs": 50,
        "warmup_runs": 10,
        "assert_tolerance": 1e-5,
    },
    "gpt2": {
        "type": "llm",
        "measure_runs": 5,  # LLM은 느리므로 5회만 측정
        "warmup_runs": 2,
        "assert_tolerance": 1e-3, # 최적화 시 오차 허용 범위가 더 큼
    },
    "distilbert": {
        "type": "llm_encoder",
        "model_id_for_load": "distilbert-base-uncased",
        "measure_runs": 10,
        "warmup_runs": 5,
        "assert_tolerance": 1e-3, # LLM 계열은 오차 허용
    },
    "clip_text": {
        "type": "clip_text",
        "model_id_for_load": "openai/clip-vit-base-patch32", 
        "measure_runs": 30,
        "warmup_runs": 5,
        "assert_tolerance": 1e-3,
    },
    "efficientnet_b0": {
        "type": "cv", 
        "measure_runs": 50,
        "warmup_runs": 10,
        "assert_tolerance": 1e-5,
    },
    "segformer_b0": {
        "type": "cv_segmentation",
        "model_id_for_load": "nvidia/segformer-b0-finetuned-ade-512-512",
        "measure_runs": 10, 
        "warmup_runs": 2,
        "assert_tolerance": 1e-3,
    },
}

def pytest_addoption(parser):
    """ 'pytest --model <name>' 커스텀 명령어를 pytest에 추가. """
    parser.addoption(
        "--model",
        action="store",
        required=True, 
        choices=MODEL_CONFIGS.keys(), 
        help="Select the model to test from: " + ", ".join(MODEL_CONFIGS.keys()),
    )

@pytest.fixture(scope="session")
def model_config(request):
    """
    CLI에서 --model로 선택된 모델의 이름과 설정을 읽어오는 Fixture
    이 Fixture가 테스트 함수에 설정값을 주입.
    """
    model_name = request.config.getoption("--model")
    config = MODEL_CONFIGS[model_name]
    
    # 모델 이름과 설정을 딕셔너리로 반환
    return {"name": model_name, **config}
