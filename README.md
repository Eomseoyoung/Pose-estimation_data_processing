# 포즈 기반 행동 해석 프로젝트

이 프로젝트는 2D 포즈 시퀀스 데이터를 기반으로 사람의 행동을 해석하는 파이프라인을 구축합니다. **상태 분류 → 행동 분류 → 행동 예측**의 3단계 구조를 목표로 하며, 현재는 1단계(상태 분류)가 구현되어 있습니다.

---

## 프로젝트 구조

```
Pose_estimation/
├── config.yaml               # 프로젝트 설정 파일
├── data/
│   ├── Raw/                  # 원본 데이터 (CSV)
│   ├── Processed/            # 정규화된 데이터 (CSV)
│   └── Train_data.h5         # 학습용 H5 데이터
├── models/                   # 학습된 모델 가중치
├── core/
│   ├── State_classifier.py   # Transformer 모델 정의
│   └── Trainer.py            # 학습/검증/평가 함수
├── script/
│   ├── Data_preprocess.py    # 데이터 정규화 실행
│   └── Classification_state.py # 상태 분류 학습 실행
├── utils/
│   ├── config_loader.py      # 설정 로더
│   ├── Dataloader.py         # DataLoader 및 시나리오 분할 함수
│   ├── Normalizer.py         # 포즈 정규화
│   └── Preprocess.py         # 시퀀스 생성 및 H5 저장
├── Main_train.py             # 전체 학습 파이프라인
├── Main_predict.py           # 예측 파이프라인
└── README.md
```

---

## 주요 모듈 설명

- **config.yaml**: 데이터 경로, 모델 파라미터, 시나리오 설정 등 프로젝트 전역 설정
- **utils/Dataloader.py**: H5 데이터 로딩, 시나리오 기반 데이터 분할 및 DataLoader 생성
- **utils/Normalizer.py**: 포즈 데이터 정규화 클래스 및 함수
- **utils/Preprocess.py**: 시퀀스 데이터 생성, 레이블 매핑, H5 저장
- **core/State_classifier.py**: Transformer 기반 상태 분류 모델 정의
- **core/Trainer.py**: 학습, 검증, 평가 함수 (validation 지원)
- **script/Data_preprocess.py**: CSV → 정규화 CSV 변환 실행
- **script/Classification_state.py**: 전체 상태 분류 학습 실행 (v1/v2 지원)
- **Main_train.py**: 데이터 정규화 → 시퀀스 생성 → 학습 전체 파이프라인
- **Main_predict.py**: 학습된 모델로 새로운 데이터 예측

---

## 파이프라인 요약

1. **데이터 정규화**  
    - `script/Data_preprocess.py` 및 `utils/Normalizer.py` 사용
    - Raw 데이터 → 정규화된 CSV

2. **시퀀스 및 H5 생성**  
    - `utils/Preprocess.py`로 시퀀스 분할, 레이블 매핑
    - H5 파일로 저장

3. **DataLoader 및 시나리오 분할**  
    - `utils/Dataloader.py`에서 H5 로딩, 시나리오별 train/val/test 분할 지원

4. **모델 학습**  
    - `core/State_classifier.py`의 Transformer 모델
    - `core/Trainer.py`의 train_model, train_model_v2 (validation 포함)
    - best model 자동 저장

5. **예측**  
    - `Main_predict.py`에서 정규화 → 시퀀스 변환 → 예측 → 결과 출력

---

## 실행 방법


---

## v1 / v2 학습 모드 차이점

이 프로젝트의 상태 분류 학습 파이프라인은 **v1**과 **v2** 두 가지 모드를 지원합니다.

- **v1 (기본 모드)**
    - 단일 CSV 또는 여러 CSV 파일을 단일 데이터셋으로 취급하여 전체 데이터를 한 번에 정규화 및 시퀀스화합니다.
    - 검증 데이터 분할 없이 전체 데이터를 학습에 사용합니다.
    - 현재 제공된 예시 데이터(`openpose_person_1.csv` 등)는 v1 방식에 적합합니다.

- **v2 (시나리오 기반 모드)**
    - `data/Raw/` 하위에 여러 개의 시나리오 폴더(예: `scenario1`, `scenario2`, ...)를 생성하고, 각 폴더에 여러 CSV 파일을 배치합니다.
    - 각 시나리오 폴더를 기준으로 데이터를 **train/validation/test** 세트로 자동 분할하여 학습 및 검증을 수행합니다.
    - 실험적/확장적 연구나 실제 환경에서의 일반화 성능 평가에 적합합니다.
    - **현재는 v2에 해당하는 시나리오 폴더 구조의 데이터가 준비되어 있지 않아, v1 모드만 정상적으로 동작합니다.**

> ⚠️ v2 모드로 학습을 시도할 경우, 시나리오별 폴더와 충분한 CSV 데이터가 필요합니다. 데이터가 없을 경우 에러가 발생할 수 있습니다.

---

### 1. 학습

```bash
python Main_train.py
```
- config.yaml에서 경로, 파라미터 등 설정
- Raw 데이터 준비 → 정규화 → 시퀀스/H5 생성 → 학습/검증/모델 저장

### 2. 예측

```bash
python Main_predict.py
```
- 예측할 CSV를 data/Raw/에 위치
- 학습된 모델로 상태 예측 결과 출력

---

## 요구 사항

- Python 3.x
- PyTorch
- pandas, numpy, h5py, pyyaml

```bash
pip install torch pandas numpy h5py pyyaml
```

---

## 참고

- 각 단계별 세부 옵션 및 시나리오 분할은 config.yaml에서 관리
- 향후 행동 분류/예측(2, 3단계) 확장 예정

---
- **`model2-2` (앉아있을 때)**: 서기, 눕기, 앉아있기(same) 등
- **`model2-3` (누워있을 때)**: 구르기, 일어서기, 앉기, 누워있기(same) 등

### 3단계: 행동 예측 (Action Prediction) - `model3` (구현 예정)
- 과거의 상태 및 행동 시퀀스를 바탕으로 미래에 발생할 행동을 예측합니다.

---

## 요구 사항

- Python 3.x
- PyTorch
- pandas
- numpy
- PyYAML
- h5py

## 설치

필요한 라이브러리를 설치합니다.

```bash
pip install torch pandas numpy pyyaml h5py
```

---

## 사용법 (현재 `model1` 기준)

모든 설정은 `config.yaml` 파일에서 관리됩니다. 데이터 경로, 모델 파라미터 등을 필요에 맞게 수정한 후 아래 스크립트를 실행하세요.

### **1. 모델 학습 (`Main_train.py`)**

`Main_train.py` 스크립트는 데이터 정규화부터 모델 학습까지 전체 파이프라인을 순차적으로 실행합니다.
# v1의 경우
1.  **데이터 준비**: `data/Raw/` 디렉터리에 학습에 사용할 원본 포즈 데이터(예: `openpose_person_1.csv`)를 위치시킵니다. CSV 파일에는 각 프레임의 포즈 키포인트 좌표와 함께 'state' 컬럼이 포함되어야 합니다.
# v2의 경우
1.  **데이터 준비**: `data/Raw/` 디렉터리에 Scenario 타입 (현재 8개) 별로 폴더를 만들어서 15개씩 데이터를 배당합니다. CSV 파일에는 각 프레임의 포즈 키포인트 좌표와 함께 'state' 컬럼이 포함되어야 합니다.
    

2.  **설정 확인**: `config.yaml` 파일에서 데이터 경로, 모델 하이퍼파라미터, 학습 옵션을 확인하고 필요시 수정합니다.

3.  **학습 실행**: 아래 명령어로 학습 파이프라인을 시작합니다.

    ```bash
    python Main_train.py
    ```

#### **학습 파이프라인 상세 과정**

`Main_train.py`를 실행하면 다음 과정이 순서대로 진행됩니다.

1.  **데이터 정규화 (`script/Data_preprocess.py`)**
    -   `data/Raw/`에 있는 원본 CSV 파일을 읽어옵니다.
    -   `utils/Normalizer.py`의 `PoseNormalizer`를 사용하여 포즈 데이터를 정규화합니다.
        -   **좌표 정규화**: 몸통 길이를 기준으로 스케일을 맞추고, 골반 중앙을 원점으로 이동시킵니다.
        -   **Y축 반전**: 이미지 좌표계를 데카르트 좌표계로 변환합니다.
    -   정규화된 데이터는 `data/Processed/` 디렉터리에 새로운 CSV 파일로 저장됩니다.

2.  **학습 데이터셋 생성 (`script/Classification_state.py` -> `utils/Preprocess.py`)**
    -   정규화된 CSV 파일을 다시 읽어옵니다.
    -   `config.yaml`에 정의된 `seq_length`에 맞춰 데이터를 시퀀스로 분할합니다.
    -   각 시퀀스에 해당하는 레이블('state')을 할당합니다.
    -   생성된 시퀀스 데이터(X)와 레이블(y)을 `data/Train_data.h5` 파일로 저장하여 학습 시 빠르게 로드할 수 있도록 합니다.

 // 추후 1-2단계 통합 가능성 높음 // 

3.  **모델 학습 (`script/Classification_state.py` -> `core/Trainer.py`)**
    -   `utils/Dataloader.py`를 사용해 `Train_data.h5` 파일에서 학습 데이터를 배치 단위로 로드하는 `DataLoader`를 생성합니다.
    -   `config.yaml`의 하이퍼파라미터를 바탕으로 `core/State_classifier.py`에 정의된 `PoseClassifierTransformer` 모델을 초기화합니다.
    -   `core/Trainer.py`의 `train_model` 함수를 호출하여 지정된 `num_epochs`만큼 학습을 진행합니다.
    -   학습이 완료되면, 학습된 모델의 가중치가 `models/state_classifier_model.pth` 파일로 저장됩니다.




### **2. 예측 (`Main_predict.py`)**

학습된 모델을 사용하여 새로운 데이터에 대한 상태를 예측합니다. (현재 미검증 상태)

1.  **데이터 준비**: 예측할 데이터(예: `predict_data.csv`)를 `data/Raw/` 디렉터리에 위치시킵니다.
2.  **예측 실행**: 아래 명령어를 실행합니다.

    ```bash
    python Main_predict.py
    ```
    -   스크립트는 `models/state_classifier_model.pth` 가중치를 로드하고, 입력 데이터를 정규화한 뒤 상태를 예측하여 결과를 출력합니다.