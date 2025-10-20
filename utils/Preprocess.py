"""
포즈 시퀀스 데이터 전처리 및 저장을 위한 스크립트.

이 스크립트는 다음 단계를 수행합니다:
1. CSV 파일에서 포즈 데이터를 로드합니다.
2. `PoseNormalizer`를 사용하여 데이터를 정규화합니다.
3. 모델 학습에 필요한 특정 관절(keypoints)만 선택합니다.
4. 데이터를 고정된 길이(sequence length)의 시퀀스로 분할합니다.
5. 생성된 시퀀스 데이터(X)와 해당 레이블(y)을 HDF5(.h5) 파일로 저장합니다.
"""
import pandas as pd
import numpy as np
import h5py
import os
from utils.Normalizer import PoseNormalizer



# 상태 문자열을 정수 레이블로 매핑 (실제 데이터에 맞게 수정 필요)
STATE_MAP = {
    'stand': 0,
    'sit': 1,
    'lying': 2,
}

def create_sequences(data: np.ndarray, seq_length: int) -> np.ndarray:
    """연속적인 데이터에서 시퀀스를 생성합니다."""
    sequences = []
    # len(data) - seq_length + 1 만큼의 시퀀스를 생성할 수 있음
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def generate_sequences(
    df: pd.DataFrame, keypoint_indices: list[int], seq_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    정규화된 DataFrame에서 시퀀스 데이터와 레이블을 생성합니다.
    'state' 컬럼을 사용하여 각 시퀀스의 레이블을 결정합니다.

    Args:
        df (pd.DataFrame): 정규화된 포즈 데이터프레임.
        keypoint_indices (list[int]): 사용할 관절 인덱스 리스트.
        seq_length (int): 생성할 시퀀스의 길이 (시간 축 길이).

    Returns:
        (sequences, labels) 형태의 튜플.
    """
    # 1. 사용할 관절 컬럼 이름 생성 ('5_x', '5_y', '6_x', ...)
    selected_columns = [f'{idx}_{axis}' for idx in keypoint_indices for axis in ['x', 'y']]

    # 2. 'state' 컬럼 확인 및 레이블 배열 생성
    if 'state' not in df.columns:
        raise ValueError("입력 DataFrame에 'state' 컬럼이 없습니다. 레이블을 생성할 수 없습니다.")
    
    # 문자열 레이블을 STATE_MAP을 사용해 정수 ID로 변환
    labels_array = df['state'].map(STATE_MAP).to_numpy()

    # 3. 주요 관절 선택 및 결측치 처리
    pose_data = df.reindex(columns=selected_columns).fillna(0.0)

    # 4. 시퀀스 및 해당 레이블 생성
    sequences = []
    labels = []
    for i in range(len(pose_data) - seq_length + 1):
        sequences.append(pose_data.values[i:i + seq_length])
        # 시퀀스의 마지막 프레임의 레이블을 해당 시퀀스의 레이블로 사용
        labels.append(labels_array[i + seq_length - 1])

    return np.array(sequences), np.array(labels, dtype=int)

def preprocess_and_save(
    input_files: list[str], output_path: str, seq_length: int, keypoint_indices: list[int]
):
    """
    여러 CSV 파일을 처리하고, 시퀀스를 생성하여 H5 파일로 저장합니다.

    Args:
        input_files (list): 처리할 CSV 파일 경로의 리스트.
        output_path (str): 저장할 H5 파일 경로.
        seq_length (int): 각 시퀀스의 프레임 길이.
        keypoint_indices (list): 사용할 관절의 인덱스 리스트.
    """
    if keypoint_indices is None:
        KeyError("keypoint_indices가 None입니다. 사용할 관절 인덱스를 지정해야 합니다.")

    all_sequences = []
    all_labels = []

    for csv_path in input_files:
        print(f"처리 중: '{csv_path}'")

        # 1. 이미 정규화된 CSV 파일 읽기
        normalized_df = pd.read_csv(csv_path, index_col=0)
        # 2. 시퀀스 및 레이블 생성
        sequences, labels = generate_sequences(normalized_df, keypoint_indices, seq_length)
        if sequences.shape[0] > 0:
            all_sequences.append(sequences)
            all_labels.append(labels)

    if not all_sequences:
        print("생성된 시퀀스가 없습니다. 입력 파일이나 시퀀스 길이를 확인해주세요.")
        return

    # 5. 모든 데이터를 하나로 합치고 H5 파일로 저장
    X_data = np.concatenate(all_sequences, axis=0)
    y_data = np.concatenate(all_labels, axis=0)

    print(f"\n총 {len(input_files)}개 파일로부터 {X_data.shape[0]}개의 시퀀스 생성 완료.")
    print(f"X_data 형태: {X_data.shape}") # (num_sequences, seq_length, num_features)
    print(f"y_data 형태: {y_data.shape}") # (num_sequences,)

    # H5 파일을 저장하기 전에 디렉터리가 존재하는지 확인하고, 없으면 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('x_data', data=X_data)
        hf.create_dataset('y_data', data=y_data)

    print(f"\n데이터를 '{output_path}' 파일로 성공적으로 저장했습니다.")
