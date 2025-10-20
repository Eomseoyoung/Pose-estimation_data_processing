import pandas as pd
import numpy as np
from utils.config_loader import CONFIG



class PoseNormalizer:
    """
    실시간(Live-stream) 및 일괄(Batch) 처리를 위한 포즈 정규화 클래스.

    실시간 처리 시, 프레임 간의 상태(단위 길이, 엉덩이 중심)를 유지하여
    결측치가 발생하더라도 연속적인 정규화를 수행합니다.
    """
    def __init__(self, initial_unit_length: float = None, initial_mid_hip: tuple = None):
        """
        Args:
            initial_unit_length (float, optional): 스트림 시작 시 사용할 초기 단위 길이.
            initial_mid_hip (tuple, optional): (x, y) 형태의 초기 엉덩이 중심 좌표.
        """
        COCO_KEYPOINTS = CONFIG.get('COCO_KEYPOINTS', None)
        KEYPOINT_NAME_TO_INDEX_MAP = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

        self.keypoint_map = KEYPOINT_NAME_TO_INDEX_MAP
        self.coco_keypoints = COCO_KEYPOINTS
        self.last_valid_unit_length = initial_unit_length
        self.last_valid_mid_hip = initial_mid_hip  # (x, y) tuple

    def _normalize_column_names(self, data):
        """컬럼 이름을 'nose_x' -> '0_x' 형태로 변환합니다."""
        rename_dict = {}
        # DataFrame의 컬럼 또는 Series의 인덱스를 순회
        for col in data.index if isinstance(data, pd.Series) else data.columns:
            if col.endswith('_x'):
                name, axis = col[:-2], 'x'
            elif col.endswith('_y'):
                name, axis = col[:-2], 'y'
            else:
                continue

            if name in self.keypoint_map:
                rename_dict[col] = f"{self.keypoint_map[name]}_{axis}"
        
        if isinstance(data, pd.DataFrame):
            # DataFrame의 경우, 'columns' 인자를 명시해야 컬럼 이름이 변경됩니다.
            return data.rename(columns=rename_dict)
        return data.rename(rename_dict) # Series의 경우, 기본적으로 인덱스를 변경합니다.

    def process_live_frame(self, frame_data: pd.Series) -> pd.Series:
        """단일 프레임 데이터를 실시간으로 정규화합니다."""
        # --- 컬럼 이름 정규화 ---
        keypoints = self._normalize_column_names(frame_data.copy())

        # --- Step 1: 데이터 무결성 적용 ---
        keypoints.replace(0.0, np.nan, inplace=True)

        # --- Step 2: 단위 길이 계산 ---
        current_unit_length = calculate_unit_length(keypoints)
        if pd.notna(current_unit_length):
            self.last_valid_unit_length = current_unit_length
        
        unit_length = self.last_valid_unit_length
        if unit_length is None or unit_length < 1e-6:
            # 유효한 단위 길이가 아직 없으면 정규화 불가
            keypoints[:] = np.nan
            return keypoints

        # --- Step 3: 단위 길이 기반 좌표 정규화 ---
        # 현재 프레임의 엉덩이 중심 계산
        current_mid_hip_x = (keypoints.get('11_x', np.nan) + keypoints.get('12_x', np.nan)) / 2
        current_mid_hip_y = (keypoints.get('11_y', np.nan) + keypoints.get('12_y', np.nan)) / 2

        if pd.notna(current_mid_hip_x) and pd.notna(current_mid_hip_y):
            self.last_valid_mid_hip = (current_mid_hip_x, current_mid_hip_y)

        mid_hip = self.last_valid_mid_hip
        if mid_hip is None:
            # 유효한 엉덩이 중심이 아직 없으면 정규화 불가
            keypoints[:] = np.nan
            return keypoints

        # (좌표 - 원점) / 단위길이
        coord_cols = keypoints.index.str.contains('_x$|_y$', regex=True)
        for col in keypoints.index[coord_cols]:
            axis = col.split('_')[-1]
            origin = mid_hip[0] if axis == 'x' else mid_hip[1]
            keypoints[col] = (keypoints[col] - origin) / unit_length

        # --- Step 4: 상하 반전 (y축 반전) ---
        y_cols = keypoints.index.str.endswith('_y')
        keypoints.loc[y_cols] *= -1

        return keypoints

    def process_csv_batch(self, filepath: str) -> pd.DataFrame:
        """CSV 파일을 읽어 일괄적으로 정규화합니다."""
        df = pd.read_csv(filepath, index_col=0)
        df = self._normalize_column_names(df)

        # --- Step 1: 데이터 무결성 적용 ---
        df.replace(0.0, np.nan, inplace=True)

        # --- Step 2: 단위 길이 계산 ---
        df['unit_length'] = calculate_unit_length(df)
        df['unit_length'].fillna(method='ffill', inplace=True)
        df['unit_length'].fillna(method='bfill', inplace=True)
        df['unit_length'] = df['unit_length'].clip(lower=1e-6)

        # --- Step 3: 단위 길이 기반 좌표 정규화 ---
        coord_cols = [col for col in df.columns if '_x' in col or '_y' in col]
        
        # COCO 기준: left_hip(11), right_hip(12)
        df['mid_hip_x'] = (df['11_x'] + df['12_x']) / 2
        df['mid_hip_y'] = (df['11_y'] + df['12_y']) / 2
        df['mid_hip_x'].fillna(method='ffill', inplace=True)
        df['mid_hip_y'].fillna(method='ffill', inplace=True)
        df['mid_hip_x'].fillna(method='bfill', inplace=True)
        df['mid_hip_y'].fillna(method='bfill', inplace=True)

        for col in coord_cols:
            axis = col.split('_')[-1]
            df[col] = (df[col] - df[f'mid_hip_{axis}']) / df['unit_length']

        # --- Step 4: 상하 반전 (y축 반전) ---
        y_cols = [col for col in df.columns if '_y' in col]
        df[y_cols] *= -1

        df.drop(columns=['unit_length', 'mid_hip_x', 'mid_hip_y'], inplace=True)
        return df


def simulate_live_from_csv(normalizer: PoseNormalizer, input_path: str, output_path: str):
    """(기존 Live 모드) CSV 파일을 한 줄씩 읽어 실시간 스트림을 시뮬레이션합니다."""
    print("--- [라이브 스트림 모드] CSV 기반 시뮬레이션 시작 ---")
    raw_df = pd.read_csv(input_path, index_col=0)
    livestream_results = []
    
    # 각 row(프레임)를 순회하며 실시간 처리 메서드 호출
    for frame_index, frame_data in raw_df.iterrows():
        # process_live_frame은 한 번에 한 프레임의 데이터(pd.Series)를 처리합니다.
        normalized_frame = normalizer.process_live_frame(frame_data)
        livestream_results.append(normalized_frame)

    livestream_df = pd.DataFrame(livestream_results, index=raw_df.index)
    print("실시간 스트림 처리 시뮬레이션 완료!")
    print(livestream_df.head())
    livestream_df.to_csv(output_path)
    print(f"결과를 '{output_path}' 파일로 저장했습니다.")


def hypothetical_video_loop():
    """
    실제 비디오 스트림을 처리하는 루프의 가상 예시입니다.
    이 함수는 직접 실행되지 않으며, 코드 통합 방법을 보여주기 위한 개념적 예제입니다.
    실제 구현 시에는 `your_pose_model`과 `cv2` 같은 라이브러리가 필요합니다.
    """
    # from your_pose_model import extract_pose_from_frame
    # import cv2 

    normalizer = PoseNormalizer()
    # video_capture = cv2.VideoCapture(0) # 0번 카메라 또는 "video.mp4"

    # while video_capture.isOpened():
    #     success, frame = video_capture.read()
    #     if not success:
    #         break

    #     # 1. 비디오 프레임에서 포즈 좌표(pd.Series)를 추출합니다.
    #     raw_pose_data = extract_pose_from_frame(frame) 
    #     # 2. 추출된 좌표를 정규화합니다.
    #     normalized_pose = normalizer.process_live_frame(raw_pose_data)
    #     # 3. 정규화된 데이터를 사용합니다 (예: 동작 분류 모델의 입력으로 사용).
    #     print("Normalized Pose:", normalized_pose)

def calculate_unit_length(data):
    """
    몸통 길이를 기반으로 단위 길이를 계산합니다.
    단위 길이 = (어깨 중앙-엉덩이 중앙 거리) + (엉덩이 중앙-무릎 중앙 거리)
    필요한 관절 좌표 중 하나라도 NaN이면, 계산 결과도 NaN이 됩니다.

    Args:
        data (pd.DataFrame or pd.Series): 키포인트 좌표 데이터.
            - DataFrame: 여러 프레임의 데이터.
            - Series: 단일 프레임의 데이터.

    Returns:
        pd.Series or float: 계산된 단위 길이. DataFrame 입력 시 Series, Series 입력 시 float 반환.
    """
    # COCO 기준: left_shoulder(5), right_shoulder(6), left_hip(11), right_hip(12), left_knee(13), right_knee(14)
    # 입력 데이터가 Series일 경우, .get()을 사용하여 안전하게 값에 접근
    mid_shoulder = (np.array([data.get('5_x'), data.get('5_y')]) + np.array([data.get('6_x'), data.get('6_y')])) / 2
    mid_hip = (np.array([data.get('11_x'), data.get('11_y')]) + np.array([data.get('12_x'), data.get('12_y')])) / 2
    mid_knee = (np.array([data.get('13_x'), data.get('13_y')]) + np.array([data.get('14_x'), data.get('14_y')])) / 2

    # 중앙점 간의 유클리드 거리 계산
    dist_shoulder_hip = np.linalg.norm(mid_shoulder - mid_hip, axis=0)
    dist_hip_knee = np.linalg.norm(mid_hip - mid_knee, axis=0)
    
    # 단위 길이 계산 (두 거리의 합)
    unit_length = dist_shoulder_hip + dist_hip_knee
    return pd.Series(unit_length, index=data.index) if isinstance(data, pd.DataFrame) else unit_length