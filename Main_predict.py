"""
학습된 상태 분류 모델을 사용하여 새로운 포즈 데이터에 대한 예측을 수행합니다.
아직 검증되지 않았습니다. (미구현)


워크플로우:
1. 예측할 원본 데이터(CSV)를 지정합니다.
2. 데이터를 정규화합니다.
3. 학습 시와 동일한 모델 구조를 생성하고, 저장된 가중치를 불러옵니다.
4. 정규화된 데이터를 모델 입력 형식에 맞게 시퀀스로 변환합니다.
5. 모델을 사용하여 각 시퀀스의 상태를 예측하고 결과를 출력합니다.
"""
import torch
import pandas as pd
import numpy as np
import os
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import CONFIG
from core.State_classifier import PoseClassifierTransformer
from utils.Normalizer import PoseNormalizer
from utils.Preprocess import generate_sequences, STATE_MAP

def run_prediction(PREDICT_RAW_CSV: str = None, train_version: str = None):
    """
    학습된 모델로 상태 예측을 수행합니다.
    
    Args:
        PREDICT_RAW_CSV (str): 예측을 수행할 원본 CSV 파일 경로.
        train_version (str, optional): 불러올 모델의 버전. Defaults to None.
    """
    # --- 예측할 데이터 및 모델 경로 ---
    # model1 (State_Classifier)에 대한 설정을 가져옵니다.
    model_cfg = CONFIG['models']['model1']
    paths_cfg = CONFIG['data_paths']

    if train_version:
        MODEL_PATH = os.path.join(paths_cfg['model_dir'], train_version, 'state_classifier_model.pth')
        print(f"--- 버전 '{train_version}'의 모델을 사용합니다. ---")
    else:
        MODEL_PATH = os.path.join(paths_cfg['model_dir'], 'state_classifier_model.pth')
        print("--- 기본 모델을 사용합니다. ---")

    # --- 파이프라인 1: 예측할 데이터 정규화 ---
    print("="*50)
    print(f"--- 예측 데이터 정규화 시작: '{PREDICT_RAW_CSV}' ---")
    if not os.path.exists(PREDICT_RAW_CSV):
        print(f"오류: 예측할 파일 '{PREDICT_RAW_CSV}'을(를) 찾을 수 없습니다.")
        return
    
    normalizer = PoseNormalizer()
    normalized_df = normalizer.process_csv_batch(PREDICT_RAW_CSV)
    print("--- 예측 데이터 정규화 완료 ---")

    # --- 파이프라인 2: 모델 로드 ---
    print("="*50)
    print(f"--- 모델 로드 시작: '{MODEL_PATH}' ---")
    if not os.path.exists(MODEL_PATH):
        print(f"오류: 학습된 모델 파일 '{MODEL_PATH}'을(를) 찾을 수 없습니다.")
        print("먼저 'Main_train.py'를 실행하여 모델을 학습하고 저장해주세요.")
        return

    model = PoseClassifierTransformer(
        input_dim=model_cfg['architecture']['input_dim'],
        d_model=model_cfg['architecture']['d_model'],
        nhead=model_cfg['architecture']['n_head'],
        num_encoder_layers=model_cfg['architecture']['num_layers'],
        dim_feedforward=model_cfg['architecture']['ffn_dim'],
        num_classes=model_cfg['architecture']['num_classes']
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # 모델을 추론 모드로 설정
    print("--- 모델 로드 완료 ---")

    # --- 파이프라인 3: 데이터 전처리 및 예측 ---
    print("="*50)
    print("--- 예측 실행 ---")
    
    has_state_column = 'state' in normalized_df.columns

    if not has_state_column:
        print("경고: 'state' 컬럼이 없어 정확도를 계산할 수 없습니다. 임시 레이블을 사용합니다.")
        normalized_df['state'] = 'stand' # 임시 레이블

    # 시퀀스 생성
    sequences, labels = generate_sequences(
        normalized_df, 
        model_cfg['data']['keypoint_indices'], 
        model_cfg['data']['seq_length']
    )
    if sequences.shape[0] == 0:
        print("예측을 위한 시퀀스를 생성할 수 없습니다. 데이터가 너무 짧습니다.")
        return

    X_predict = torch.tensor(sequences, dtype=torch.float32)

    # 예측 수행
    with torch.no_grad(): # 그래디언트 계산 비활성화
        outputs = model(X_predict)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_indices = torch.argmax(probabilities, dim=1)

    # 결과를 사람이 읽을 수 있는 형태로 변환
    idx_to_state = {v: k for k, v in STATE_MAP.items()}
    predicted_labels = [idx_to_state[idx.item()] for idx in predicted_indices]
    result_df = pd.DataFrame()
    # 1) header: frame을 생성해서 값을 채워넣을 것
    result_df['frame'] = np.arange(len(normalized_df))+1
    
    # 3) header: labeled_state를 만들어서 기존 normalized_df['state'] 값을 가져올 것
    result_df['labeled_state'] = normalized_df['state'].reset_index(drop=True)
    
    # 2) model_config/data/seq_length 값을 바탕으로 predict 값을 해당 위치부터 채워넣을 것
    seq_length = model_cfg['data']['seq_length']
    prediction_start_index = seq_length - 1
    
    # 예측 결과를 담을 'predict-state' 컬럼을 생성하고 object 타입으로 초기화
    result_df['predict-state'] = pd.Series(dtype='object')
    
    # 예측된 라벨을 'predict-state' 컬럼의 해당 위치에 할당
    # 예측된 라벨의 수는 len(normalized_df) - seq_length + 1 입니다.
    # 따라서 prediction_start_index 부터 끝까지 할당합니다.
    end_index = prediction_start_index + len(predicted_labels)
    result_df.loc[prediction_start_index:end_index-1, 'predict-state'] = predicted_labels

    # 결과 저장
    result_csv_path = os.path.join(paths_cfg['results_dir'], 'predicted_states_aligned.csv')
    result_df.to_csv(result_csv_path, index=False)
    print(f"정렬된 예측 결과를 '{result_csv_path}' 파일로 저장했습니다.")

    # 4) 정확도 계산을 둘 다 존재하는 값만 비교해서 % 낼 것.
    if has_state_column:
        # 'labeled_state'와 'predict-state' 둘 다 유효한 값만 필터링
        valid_comparison_df = result_df.dropna(subset=['labeled_state', 'predict-state'])
        
        if not valid_comparison_df.empty:
            correct_predictions = (valid_comparison_df['labeled_state'] == valid_comparison_df['predict-state']).sum()
            total_predictions = len(valid_comparison_df)
            
            if total_predictions > 0:
                accuracy = (correct_predictions / total_predictions) * 100
                print(f"\n--- 예측 정확도 (정렬 후): {accuracy:.2f}% (정답 {correct_predictions} / 전체 {total_predictions}) ---")
        else:
            print("\n정확도를 계산할 수 있는 유효한 데이터가 없습니다.")

    print(f"\n총 {len(predicted_labels)}개의 시퀀스에 대한 예측이 수행되었습니다.")


if __name__ == '__main__':
    
    # --- 예측 실행 ---
    # 예측할 파일과 모델 버전을 설정합니다.
    # train_version에 'v1', 'v2' 등 학습 시 사용한 버전을 입력합니다.
    # None으로 두면 'models/state_classifier_model.pth'의 기본 모델을 사용합니다.
    
    PREDICT_RAW_CSV = 'openpose_person_1.csv'
    # TRAIN_VERSION = 'v1, v2' 
    TRAIN_VERSION = None
    # predictor는 버전에 따라 불러오는 모델 경로만 다름
    run_prediction(PREDICT_RAW_CSV=PREDICT_RAW_CSV, train_version=TRAIN_VERSION)
