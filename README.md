이 프로젝트는 원본 PyTorch 모델과 ExecuTorch(.pte)로 변환 및 최적화된 모델 간의 수치적 동일성(정확도)과 추론 성능(Latency)을 비교 검증하는 자동화된 도구입니다.
다양한 종류의 딥러닝 모델(CV, LLM 등)을 지원하며, 사용자는 간단한 GUI 또는 CLI (pytest) 인터페이스를 통해 원하는 모델을 선택하여 검증을 수행할 수 있습니다.

### 구현 기능 요약

- **다양한 모델 지원 및 동적 로딩:**
  - `conftest.py` 설정을 통해 여러 모델 아키텍처 지원:
    - **CV (CNN):** `resnet18`, `mobilenet_v2`, `efficientnet_b0` (`torchvision` 사용)
    - **LLM (Decoder):** `gpt2` (`transformers` 사용)
    - **LLM (Encoder):** `distilbert` (`transformers` 사용)
    - **Multimodal Component:** `clip_text` (`transformers` 사용)
    - **CV (Segmentation):** `segformer_b0` (`transformers` 사용)
  - `pytest` 실행 또는 GUI 버튼 클릭 시 선택된 모델을 동적으로 로드합니다.
- **모델별 설정 관리:**
  - `conftest.py`에서 모델별 측정 횟수(`measure_runs`), 예열 횟수(`warmup_runs`), 오차 허용 범위(`assert_tolerance`), 실제 로드 ID(`model_id_for_load`) 등을 중앙 관리합니다.
- **자동 ExecuTorch 변환 및 최적화:**
  - 선택된 PyTorch 모델을 `torch.export`를 사용하여 Export하고, `XnnpackPartitioner`를 통해 CPU 추론에 최적화된 `.pte` 파일을 자동으로 생성합니다.
- **성능 (Latency) 비교:**
  - PyTorch 원본 모델과 ExecuTorch 변환 모델의 추론 시간을 각각 지정된 횟수만큼 실행하여 측정합니다 (`measure_latency` 함수).
  - 평균(avg) 및 최대(max) Latency를 `ms` 단위로 계산하여 비교합니다.
- **정확도 (Numerical Equivalence) 비교:**
  - 두 모델의 출력 텐서를 비교하여 평균 절대 오차(Mean Absolute Difference, MAE)를 계산합니다.
  - `conftest.py`에 정의된 모델별 `assert_tolerance` 값 미만일 경우 테스트가 통과(PASSED)하도록 자동 검증합니다 (`assert`).
- **듀얼 인터페이스 (GUI & CLI):**
  - **GUI (`verifier_gui.py`):**
    - `tkinter` 기반의 사용자 친화적 인터페이스 제공.
    - 버튼 클릭으로 간편하게 모델 테스트 실행.
    - 테스트 진행 상태 및 `pytest` 로그 실시간 표시.
    - 최종 결과(MAE, Latency 등)를 파싱하여 보기 쉽게 표시.
  - **CLI (`pytest test.py`):**
    - `pytest --model <model_name>` 명령어로 특정 모델 테스트 실행.
    - 자동화된 테스트 환경 및 스크립팅에 용이.
- **자동 리소스 정리:**
  - 테스트 종료 후 생성된 `.pte` 파일을 자동으로 삭제합니다.
  - GUI 실행 중 파일 잠금 에러 방지를 위해 ExecuTorch 런타임 객체를 명시적으로 해제합니다.


### 개발 환경

- **Python**: `3.10.19`
- 필수 패키지
  - torch==2.9.0
  - torchvision==0.24.0
  - executorch==1.0.0
  - pytest==8.4.2
  - numpy==2.2.6
  - transformers==4.57.1
- 실행 환경: x64 Native Tools Command Prompt for VS 2022 (관리자 권한)



### 설치 및 실행 방법

1. repo 복제

```bash
git clone https://github.com/mvg01/pytorch-executorch-model-validation.git
cd pytorch-executorch-model-validation
```

2. Visual Studio 2022와 "C++를 사용한 데스크톱 개발" 워크로드를 설치
3. Python 환경 준비 (Conda)

```bash
conda create -n exec_project python=3.10
conda activate exec_project
```

4. 라이브러리 설치 (pip)

```bash
pip install -r requirements.txt
```

5. 실행

   - x64 Native Tools Command Prompt for VS 2022 터미널 **관리자 권한**으로 실행

   - ```bash
     # CLI
     pytest test.py -v -s --model <model_name>
     ```
   - ```bash
     # GUI
     python gui.py
     ```



### 실행 결과 예시

- test.py를 통해 직접 model명 주입 후 테스트
<img width="1160" height="404" alt="image" src="https://github.com/user-attachments/assets/b05ed12b-ba14-4352-8937-c6359e4b950f" />


- gui 화면 구성
<img width="1002" height="732" alt="image" src="https://github.com/user-attachments/assets/54435577-a84a-42f0-899b-7a1f935c5409" />

- 실행 결과 화면
<img width="1002" height="732" alt="image" src="https://github.com/user-attachments/assets/00aa5a76-fec3-4ab1-a1f1-10940a08bbfa" />






