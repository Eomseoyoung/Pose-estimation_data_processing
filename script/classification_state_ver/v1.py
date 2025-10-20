"""
(v2) 시나리오 기반으로 분할된 데이터를 이용한 상태 분류 모델 학습 스크립트.

이 스크립트는 원본 데이터를 시나리오 단위로 분할하여 학습, 검증, 테스트를 수행합니다.
자가 참조 문제를 방지하고, 반복적인 무작위 서브 샘플링을 통한 교차 검증이 가능하도록 설계되었습니다.

주요 파이프라인:
1. `config.yaml`에서 v2 모델 설정 및 데이터 경로 로드.
2. `Dataloader_v2.py`의 `split_scenarios`를 사용해 시나리오를 학습/검증/테스트용으로 분할.
3. `Dataloader_v2.py`의 `create_scenario_dataloaders`를 사용해 각 세트에 대한 데이터 로더 생성.
   - 이 과정에서 `Preprocess.py`의 `preprocess_and_save`가 호출되어 각 세트별로 데이터가 전처리되고 H5 파일로 저장됩니다.
4. `core.Trainer.py`의 `train_model` 함수를 호출하여 모델 학습 및 평가 수행.
   - 학습 중 검증 손실을 기준으로 최고의 모델 가중치를 저장합니다.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import glob

from core.State_classifier import PoseClassifierTransformer
from utils.config_loader import CONFIG
from core.Trainer import train_model
from utils.Preprocess import preprocess_and_save
from utils.Dataloader import create_dataloader


def run_state_classification_training(processed_data_dir: str,TRAIN_VERSION: str = None):
    """
    지정된 폴더의 모든 CSV 데이터를 사용하여 데이터 전처리부터 모델 학습까지 전체 파이프라인을 실행합니다.
    
    Args:
        processed_data_dir (str): 정규화된 CSV 파일들이 포함된 디렉터리 경로.
    """
    # --- 설정 로드 ---
    model_cfg = CONFIG['models']['model1']
    paths_cfg = CONFIG['data_paths']
    
    MODEL_SAVE_PATH = os.path.join(paths_cfg['model_dir'],TRAIN_VERSION, 'state_classifier_model.pth')

    # 모델 인스턴스 생성
    model = PoseClassifierTransformer(
        input_dim=len(model_cfg['data']['keypoint_indices']) * 2,
        d_model=model_cfg['architecture']['d_model'],
        nhead=model_cfg['architecture']['n_head'],
        num_encoder_layers=model_cfg['architecture']['num_layers'],
        dim_feedforward=model_cfg['architecture']['ffn_dim'],
        num_classes=model_cfg['architecture']['num_classes']
    )

    print("--- 모델 구조 ---")
    print(model)
    print("\n" + "="*50 + "\n")

    # --- 파이프라인 1: 정규화된 데이터로 시퀀스 생성 및 H5 저장 ---
    print("--- 데이터 전처리 시작 ---")
    
    # 지정된 디렉터리 및 모든 하위 디렉터리에서 CSV 파일 검색
    print(f"'{processed_data_dir}' 및 하위 디렉터리에서 CSV 파일을 검색합니다.")
    files_to_process = glob.glob(os.path.join(processed_data_dir, '**', '*.csv'), recursive=True)
    
    if not files_to_process:
        print(f"오류: '{processed_data_dir}' 디렉터리 또는 하위 디렉터리에 처리된 CSV 파일이 없습니다.")
        return

    # 데이터 불균형 확인을 위해 모든 데이터를 하나의 데이터프레임으로 로드
    print("데이터 분포 확인 중...")
    df_list = [pd.read_csv(f) for f in files_to_process]
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"총 {len(files_to_process)}개의 CSV 파일에서 {len(full_df)}개의 프레임을 로드했습니다.")

    # 클래스별 데이터 분포 확인
    if 'state' in full_df.columns:
        print("\n--- 클래스별 데이터 분포 ---")
        print(full_df['state'].value_counts())
        print("---------------------------\n")
    else:
        print("\n경고: 'state' 컬럼이 없어 클래스 분포를 확인할 수 없습니다.\n")

    # 저장될 H5 파일 경로 지정 (data 폴더 아래)
    H5_FILE_PATH = os.path.join(paths_cfg['h5_dir'], 'Train_data.h5')

    # Preprocess.py의 유틸리티 함수를 호출하여 파일 처리 및 저장
    preprocess_and_save(
        input_files=files_to_process,
        output_path=H5_FILE_PATH,
        seq_length=model_cfg['data']['seq_length'],
        keypoint_indices=model_cfg['data']['keypoint_indices']
    )
    print("--- 데이터 전처리 완료 ---\n")
    print("="*50 + "\n")

    # --- 파이프라인 2: 학습 데이터 로드 ---
    print("--- 학습 데이터 로드 시작 ---")

    try:
        # utils/Dataloader.py의 함수를 사용하여 DataLoader 생성
        train_loader = create_dataloader(H5_FILE_PATH, model_cfg['training']['batch_size'])
        print(f"데이터 로드 완료: {len(train_loader.dataset)}개의 샘플")

    except (IOError, KeyError, ValueError) as e:
        print(f"데이터 로딩 실패: {e}")
        return

    print("--- 학습 데이터 로드 완료---\n")
    print("="*50 + "\n")

    # --- 파이프라인 3: 모델 학습 ---
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_cfg['training']['learning_rate'])

    # core/Trainer.py의 학습 함수 호출
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=model_cfg['training']['num_epochs'],
        model_save_path=MODEL_SAVE_PATH
    )

