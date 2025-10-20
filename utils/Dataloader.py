import h5py
import torch
import os
import glob
import random
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from typing import Tuple, List


def create_dataloader(h5_path: str, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    H5 파일에서 데이터를 로드하여 PyTorch DataLoader를 생성합니다.

    Args:
        h5_path (str): 데이터가 저장된 H5 파일 경로.
        batch_size (int): DataLoader의 배치 크기.
        shuffle (bool): 데이터를 섞을지 여부. 학습 시에는 True로 설정하는 것이 일반적입니다.

    Returns:
        DataLoader: 생성된 PyTorch DataLoader 객체.
    
    Raises:
        IOError, KeyError: 파일 읽기 중 오류가 발생할 경우.
        ValueError: 로드된 데이터가 비어있을 경우.
    """
    try:
        with h5py.File(h5_path, 'r') as hf:
            x_data = hf['x_data'][:]
            y_data = hf['y_data'][:]
    except (IOError, KeyError) as e:
        print(f"오류: H5 파일 '{h5_path}'을(를) 읽는 중 문제가 발생했습니다: {e}")
        raise

    if x_data.shape[0] == 0:
        raise ValueError("오류: 전처리 후 생성된 데이터가 없습니다. 학습을 진행할 수 없습니다.")

    # PyTorch 모델에 입력하기 위해 텐서로 변환
    X_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.long)  # CrossEntropyLoss는 long 타입의 레이블을 기대

    # TensorDataset과 DataLoader 생성
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def create_scenario_dataloaders(raw_data_dir: str,
                                batch_size: int,
                                train_scenarios: List[str],
                                val_scenarios: List[str],
                                test_scenarios: List[str],
                                h5_dir: str,
                                preprocess_func, # This will be preprocess_and_save from Preprocess.py
                                seq_length: int,
                                keypoint_indices: List[int]
                                ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    시나리오 기반으로 분할된 데이터셋에 대한 DataLoader를 생성합니다.

    Args:
        raw_data_dir (str): 'scenario1', 'scenario2' 등이 포함된 원본 데이터 디렉터리.
        batch_size (int): DataLoader의 배치 크기.
        train_scenarios (List[str]): 학습에 사용할 시나리오 이름 리스트.
        val_scenarios (List[str]): 검증에 사용할 시나리오 이름 리스트.
        test_scenarios (List[str]): 테스트에 사용할 시나리오 이름 리스트.
        h5_dir (str): 전처리된 H5 파일이 저장될 디렉터리.
        preprocess_func: 각 시나리오의 CSV 파일들을 전처리하고 H5로 저장하는 함수.
        seq_length (int): 시퀀스 길이.
        keypoint_indices (List[int]): 사용할 키포인트 인덱스.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (학습, 검증, 테스트) DataLoader.
    """
    
    os.makedirs(h5_dir, exist_ok=True)

    def get_dataset_for_scenarios(scenarios: List[str], set_name: str) -> TensorDataset:
        """주어진 시나리오 목록에 대해 데이터셋을 생성합니다."""
        h5_path = os.path.join(h5_dir, f'{set_name}_data.h5')
        
        # 시나리오별 CSV 파일 경로 수집
        all_files = []
        for scenario in scenarios:
            scenario_path = os.path.join(raw_data_dir, scenario)
            files = glob.glob(os.path.join(scenario_path, '**', '*.csv'), recursive=True)
            if not files:
                print(f"경고: '{scenario_path}'에서 CSV 파일을 찾을 수 없습니다.")
            all_files.extend(files)

        if not all_files:
            raise FileNotFoundError(f"{set_name} 세트를 위한 CSV 파일이 없습니다.")

        # 데이터 전처리 및 H5 파일로 저장
        preprocess_func(
            input_files=all_files,
            output_path=h5_path,
            seq_length=seq_length,
            keypoint_indices=keypoint_indices
        )

        # H5 파일에서 데이터 로드
        with h5py.File(h5_path, 'r') as hf:
            x_data = hf['x_data'][:]
            y_data = hf['y_data'][:]
        
        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.long)
        
        return TensorDataset(x_tensor, y_tensor)

    # 각 세트에 대한 데이터셋 생성
    train_dataset = get_dataset_for_scenarios(train_scenarios, 'train')
    val_dataset = get_dataset_for_scenarios(val_scenarios, 'val')
    test_dataset = get_dataset_for_scenarios(test_scenarios, 'test')

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"데이터 로더 생성 완료:")
    print(f"  - 학습용: {len(train_dataset)} 샘플")
    print(f"  - 검증용: {len(val_dataset)} 샘플")
    print(f"  - 테스트용: {len(test_dataset)} 샘플")

    return train_loader, val_loader, test_loader

def split_scenarios(raw_data_dir: str, 
                    train_ratio: float = 0.75, 
                    val_ratio: float = 0.125, 
                    seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    주어진 디렉터리에서 시나리오 폴더를 찾아 학습, 검증, 테스트 세트로 분할합니다.
    8개의 시나리오를 6:1:1 비율로 분할합니다.

    Args:
        raw_data_dir (str): 'scenario1', 'scenario2' 등이 포함된 원본 데이터 디렉터리.
        train_ratio (float): 학습 세트 비율.
        val_ratio (float): 검증 세트 비율.
        seed (int): 재현성을 위한 랜덤 시드.

    Returns:
        Tuple[List[str], List[str], List[str]]: (학습, 검증, 테스트) 시나리오 이름 리스트.
    """
    random.seed(seed)
    
    scenarios = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d)) and d.startswith('scenario')]
    if not scenarios:
        raise FileNotFoundError(f"'{raw_data_dir}'에서 'scenario'로 시작하는 폴더를 찾을 수 없습니다.")

    random.shuffle(scenarios)
    
    num_scenarios = len(scenarios)
    num_train = int(num_scenarios * train_ratio)
    num_val = int(num_scenarios * val_ratio)
    
    # 8개 시나리오 기준 6:1:1 분할 보장
    if num_scenarios == 8 and train_ratio == 0.75 and val_ratio == 0.125:
        num_train = 6
        num_val = 1

    num_test = num_scenarios - num_train - num_val
    if num_test < 1 and num_scenarios > num_train + num_val:
        num_test = 1 # 최소 1개 보장
        if num_val > 1:
            num_val -=1
        else:
            num_train -=1

    train_scenarios = scenarios[:num_train]
    val_scenarios = scenarios[num_train:num_train + num_val]
    test_scenarios = scenarios[num_train + num_val:]

    print("--- 시나리오 분할 결과 ---")
    print(f"총 {num_scenarios}개의 시나리오")
    print(f"  - 학습용: {train_scenarios}")
    print(f"  - 검증용: {val_scenarios}")
    print(f"  - 테스트용: {test_scenarios}")
    print("-------------------------")

    return train_scenarios, val_scenarios, test_scenarios
