`pytest`를 사용하여 원본 PyTorch 모델(`ResNet18`)과 ExecuTorch로 변환된 모델(`.pte`)의 추론 속도(Latency)와 결과값(정확도)을 비교하고 검증하는 자동화 테스트 스크립트입니다.


### 구현 기능 요약

- **PyTorch 모델 로드**: `torchvision`에서 `resnet18` 원본 모델을 로드합니다.
- **ExecuTorch 변환**: 로드한 모델을 `torch.export`와 `XnnpackPartitioner`를 사용해 `resnet18.pte` 파일로 자동 변환 및 최적화합니다.
- **자동화된 준비/정리**: `pytest.fixture`를 사용하여 테스트 시작 시 `.pte` 파일을 생성하고, 테스트 종료 시 자동으로 삭제합니다.
- **성능(Latency) 비교**:
  - `PyTorch` 원본 모델과 `ExecuTorch` 변환 모델의 추론 시간을 각각 50회 실행하여 측정합니다.
  - 두 결과의 평균(avg) 및 최대(max) Latency를 `ms` 단위로 출력합니다.
- **정확도(Accuracy) 비교**:
  - 두 모델의 출력 텐서를 비교하여 평균 절대 오차(Mean Absolute Difference)를 계산합니다.
- **자동화 검증**: `assert`를 사용하여 평균 절대 오차가 `1e-5` 미만일 경우에만 테스트가 통과(`PASSED`)하도록 자동화합니다.



### 개발 환경

- **Python**: `3.10.19`
- 필수 패키지
  - torch==2.9.0
  - torchvision==0.24.0
  - executorch==1.0.0
  - pytest==8.4.2
  - numpy==2.2.6
- 실행 환경: x64 Native Tools Command Prompt for VS 2022



### 설치 및 실행 방법

1. repo 복제

```bash
git clone https://github.com/mvg01/pytorch-executorch-model-validation.git
cd executorch-benchmark
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
     pytest -v -s
     ```



### 실행 결과 예시

<img width="1075" height="129" alt="image" src="https://github.com/user-attachments/assets/6fb2d961-888b-46a6-b519-e7e5e57162d7" />



