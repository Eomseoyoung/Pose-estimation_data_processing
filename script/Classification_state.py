"""
포즈 시퀀스 데이터를 이용한 상태 분류를 위한 경량 Transformer 모델.

이 스크립트는 30프레임 길이의 6개 주요 관절(12차원) 시퀀스 데이터를 입력받아,
3가지 상태 중 하나로 분류하는 Transformer 기반 모델을 정의합니다.

주요 구성 요소:
1. PositionalEncoding: 시퀀스 내 토큰의 위치 정보를 모델에 주입합니다.
2. PoseClassifierTransformer: Transformer Encoder를 기반으로 한 분류 모델.
   - 입력 임베딩: 12차원의 키포인트 데이터를 모델의 내부 차원으로 변환합니다.
   - Transformer Encoder: 시퀀스 데이터의 시간적 특징을 학습합니다.
   - 분류 헤드: Encoder의 출력을 받아 최종 클래스를 예측합니다.

모델 입력 텐서 형태: (batch_size, seq_length, input_dim)
    - batch_size: 한 번에 처리할 데이터 샘플의 수
    - seq_length: 30 (프레임 수)
    - input_dim: 12 (6개 관절 * 2D 좌표)

모델 출력 텐서 형태: (batch_size, num_classes)
    - num_classes: 3 (분류할 상태의 수)
"""
import h5py
import torch
import os
import glob
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from core.State_classifier import PoseClassifierTransformer
from utils.config_loader import CONFIG

from script.classification_state_ver import run_state_classification_training_v2, run_state_classification_training



def run_state_classification_tot(processed_data_dir: str,TRAIN_VERSION: str = None):
    if TRAIN_VERSION == 'v1' or TRAIN_VERSION is None:
        run_state_classification_training(processed_data_dir,TRAIN_VERSION) # 핸드 샘플 데이터 대상
    elif TRAIN_VERSION == 'v2':
        run_state_classification_training_v2(processed_data_dir,TRAIN_VERSION) # validation 추가, scenario 기반 data 적용
    
