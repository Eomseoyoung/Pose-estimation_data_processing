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

from core.State_classifier import PoseClassifierTransformer
from utils.config_loader import CONFIG
from core.Trainer import train_model_v2 as train_model 
from core.Trainer import evaluate_model
from utils.Preprocess import preprocess_and_save
from utils.Dataloader import split_scenarios, create_scenario_dataloaders

def run_state_classification_training_v2():
    """
    시나리오 기반 데이터 분할 및 샘플링을 사용하여 모델 학습 v2 파이프라인을 실행합니다.
    """
    print("="*50)
    print("--- 상태 분류 모델 학습 v2 (시나리오 기반) 시작 ---")
    print("="*50 + "\n")

    # --- 설정 로드 (v2 설정) ---
    try:
        model_cfg = CONFIG['models']['model1_v2']
        paths_cfg = CONFIG['data_paths']
        raw_data_dir = paths_cfg['raw_dir']
        h5_dir = os.path.join(paths_cfg['h5_dir'], 'v2') # v2용 H5 파일 저장 경로
    except KeyError as e:
        print(f"오류: config.yaml 파일에서 필요한 설정({e})을 찾을 수 없습니다.")
        return

    # --- 1. 시나리오 분할 ---
    try:
        train_scenarios, val_scenarios, test_scenarios = split_scenarios(
            raw_data_dir=raw_data_dir,
            train_ratio=model_cfg['data']['train_ratio'],
            val_ratio=model_cfg['data']['val_ratio'],
            seed=model_cfg['data']['seed']
        )
    except FileNotFoundError as e:
        print(f"오류: 시나리오 분할 중 문제가 발생했습니다. {e}")
        return

    # --- 2. 데이터 로더 생성 ---
    try:
        train_loader, val_loader, test_loader = create_scenario_dataloaders(
            raw_data_dir=raw_data_dir,
            batch_size=model_cfg['training']['batch_size'],
            train_scenarios=train_scenarios,
            val_scenarios=val_scenarios,
            test_scenarios=test_scenarios,
            h5_dir=h5_dir,
            preprocess_func=preprocess_and_save,
            seq_length=model_cfg['data']['seq_length'],
            keypoint_indices=model_cfg['data']['keypoint_indices']
        )
    except (FileNotFoundError, ValueError, IOError, KeyError) as e:
        print(f"오류: 데이터 로더 생성 중 문제가 발생했습니다. {e}")
        return

    # --- 3. 모델, 손실 함수, 옵티마이저 정의 ---
    model = PoseClassifierTransformer(
        input_dim=len(model_cfg['data']['keypoint_indices']) * 2,
        d_model=model_cfg['architecture']['d_model'],
        nhead=model_cfg['architecture']['n_head'],
        num_encoder_layers=model_cfg['architecture']['num_layers'],
        dim_feedforward=model_cfg['architecture']['ffn_dim'],
        num_classes=model_cfg['architecture']['num_classes']
    )
        # 모델 인스턴스 생성

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_cfg['training']['learning_rate'])

    # --- 4. 모델 학습 ---
    # 모델 저장 경로 설정 (v2, best model)
    MODEL_SAVE_PATH = os.path.join(paths_cfg['model_dir'], 'state_classifier_v2_best.pth')

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader, # 검증 로더 추가
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=model_cfg['training']['num_epochs'],
        model_save_path=MODEL_SAVE_PATH
    )

    # --- 5. 최종 모델 평가 ---
    # 저장된 최고 성능의 모델을 로드하여 테스트
    print("\n--- 최종 모델 평가 (테스트 세트) ---")
    best_model = PoseClassifierTransformer(
        input_dim=len(model_cfg['data']['keypoint_indices']) * 2,
        d_model=model_cfg['architecture']['d_model'],
        nhead=model_cfg['architecture']['n_head'],
        num_encoder_layers=model_cfg['architecture']['num_layers'],
        dim_feedforward=model_cfg['architecture']['ffn_dim'],
        num_classes=model_cfg['architecture']['num_classes']
    )
    try:
        best_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        evaluate_model(
            model=best_model,
            test_loader=test_loader,
            criterion=criterion
        )
    except FileNotFoundError:
        print(f"오류: 저장된 모델 파일('{MODEL_SAVE_PATH}')을 찾을 수 없어 최종 평가를 스킵합니다.")
    except Exception as e:
        print(f"오류: 최종 모델 평가 중 문제가 발생했습니다. {e}")

    print("\n" + "="*50)
    print("--- 상태 분류 모델 학습 v2 완료 ---")
    print("="*50)
