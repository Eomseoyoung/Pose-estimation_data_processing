import yaml
from typing import Any, Dict

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    YAML 설정 파일을 로드하고 파싱합니다.
    또한, 각 모델의 'input_dim'을 'keypoint_indices' 길이에 따라 동적으로 계산합니다.

    Args:
        config_path (str): YAML 설정 파일의 경로.

    Returns:
        Dict[str, Any]: 파싱 및 처리된 설정이 담긴 딕셔너리.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 동적 파라미터 계산: keypoint_indices 길이에 따라 input_dim 자동 설정
        for model_config in config.get('models', {}).values():
            if 'data' in model_config and 'keypoint_indices' in model_config['data']:
                num_keypoints = len(model_config['data']['keypoint_indices'])
                # architecture에 input_dim이 명시적으로 있어도 덮어씁니다.
                # 이는 keypoint_indices 변경 시 실수를 방지하기 위함입니다.
                model_config['architecture']['input_dim'] = num_keypoints * 2
                
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"설정 파일 '{config_path}'을(를) 찾을 수 없습니다.")
    except yaml.YAMLError as e:
        raise ValueError(f"설정 파일 '{config_path}' 파싱 중 오류 발생: {e}")

# 전역적으로 사용할 설정 객체. 스크립트 시작 시 한 번만 로드됩니다.
CONFIG = load_config()