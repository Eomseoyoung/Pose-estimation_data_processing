"""포즈 정규화 스크립트.

이 스크립트는 CSV 파일로부터 2D 포즈 추정 데이터를 정규화하는 기능을 제공합니다.
두 가지 작동 모드를 지원합니다:
1. 'batch': 전체 CSV 파일을 한 번에 처리합니다.
2. 'live': 데이터를 프레임 단위로 처리하여 실시간 스트림을 시뮬레이션합니다.

정규화 과정은 주로 4단계로 이루어집니다:
1. 데이터 무결성: 0.0 좌표를 NaN으로 대체합니다.
2. 단위 길이 계산: 몸통 길이를 기준으로 일관된 스케일을 결정합니다.
3. 좌표 정규화: 엉덩이 중앙을 원점으로 하고 단위 길이로 스케일링합니다.
4. Y축 반전: 일관된 방향을 위해 y좌표를 뒤집습니다.

사용 예시:

# 배치 처리 모드:
python Main_norm.py --mode batch --input openpose_person_1.csv --output normalized_batch.csv

# 실시간 스트림 시뮬레이션 모드:
python Main_norm.py --mode live --input openpose_person_1.csv --output normalized_live.csv
            
"""

import pandas as pd
import numpy as np
import os
import argparse
import sys
import glob
from utils.Normalizer import PoseNormalizer
from utils.Normalizer import simulate_live_from_csv

def run_normalization(input_path: str, output_path: str, mode: str = 'batch'):
    """
    주어진 경로와 모드로 포즈 데이터 정규화를 실행합니다.

    Args:
        input_path (str): 입력 CSV 파일 경로.
        output_path (str): 정규화된 결과를 저장할 CSV 파일 경로.
        mode (str): 처리 모드 ('batch' 또는 'live').
    
    Raises:
        FileNotFoundError: 입력 파일을 찾을 수 없을 때.
        KeyError: CSV에 필요한 컬럼이 없을 때.
    """
    normalizer = PoseNormalizer()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"오류: 입력 파일 '{input_path}'을(를) 찾을 수 없습니다. 경로를 확인해주세요.")

    try:
        if mode == 'batch':
            print("--- [배치 모드] CSV 파일(1개) 일괄 처리 시작 ---")
            normalized_df = normalizer.process_csv_batch(input_path)
            normalized_df.to_csv(output_path)
            print(f"일괄 처리 완료! 결과를 '{output_path}' 파일로 저장했습니다.")
        elif mode == 'live':
            simulate_live_from_csv(normalizer, input_path, output_path)

    except KeyError as e:
        print(f"오류: CSV 파일에 필요한 컬럼이 없습니다 - {e}. 컬럼명을 확인해주세요.")
        raise

def run_batch_normalization(raw_dir: str, processed_dir: str):
    """
    raw_dir 안의 모든 CSV 파일을 정규화하여 processed_dir에 저장합니다.
    하위 디렉터리까지 모두 탐색하며, 폴더 구조를 유지합니다.
    """
    normalizer = PoseNormalizer()
    
    # Ensure the output directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # Find all CSV files in the raw directory and its subdirectories
    raw_files = glob.glob(os.path.join(raw_dir, '**', '*.csv'), recursive=True)

    if not raw_files:
        print(f"경고: '{raw_dir}'에서 처리할 CSV 파일을 찾지 못했습니다.")
        return

    print(f"총 {len(raw_files)}개의 파일을 정규화합니다.")

    for raw_path in raw_files:
        try:
            # To preserve subdirectory structure, replace raw_dir with processed_dir in the path
            relative_path = os.path.relpath(raw_path, raw_dir)
            processed_path = os.path.join(processed_dir, relative_path)
            
            # Ensure the output subdirectory exists
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            
            print(f"  - 처리 중: '{raw_path}' -> '{processed_path}'")
            
            normalized_df = normalizer.process_csv_batch(raw_path)
            normalized_df.to_csv(processed_path)

        except Exception as e:
            print(f"오류: '{raw_path}' 파일 처리 중 문제 발생: {e}")
    
    print("일괄 정규화 완료.")


if __name__ == '__main__':
    # --- 스크립트 실행을 위한 인자 파서 설정 ---
    parser = argparse.ArgumentParser(
        description="실시간 또는 일괄 처리 방식으로 포즈 데이터를 정규화합니다.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='batch',
        choices=['batch', 'live'],
        help="처리 모드를 선택합니다:\n" \
             "'batch': CSV 파일을 한 번에 정규화합니다.\n" \
             "'live': CSV 파일에서 데이터를 한 줄씩 읽어 실시간처럼 처리합니다."
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="정규화를 진행할 입력 CSV 파일 경로."
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="정규화된 데이터를 저장할 출력 CSV 파일 경로."
    )

    args = parser.parse_args()

    # --- 정규화 실행 ---
    try:
        run_normalization(args.input, args.output, args.mode)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except KeyError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")
        sys.exit(1)
