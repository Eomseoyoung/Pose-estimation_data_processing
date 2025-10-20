import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np

def train_model(model: nn.Module,
                train_loader: DataLoader, 
                criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                num_epochs: int,
                model_save_path: str):
    """
    주어진 모델과 데이터로 학습을 수행합니다.

    Args:
        model (nn.Module): 학습할 PyTorch 모델.
        train_loader (DataLoader): 학습 데이터셋을 위한 DataLoader.
        criterion (nn.Module): 손실 함수.
        optimizer (torch.optim.Optimizer): 옵티마이저.
        num_epochs (int): 총 학습 에포크 수.
        model_save_path (str): 학습된 모델을 저장할 경로.
    """
    print("--- 모델 학습 시작 ---")
    model.train()  # 모델을 학습 모드로 설정

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # 옵티마이저 그래디언트 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(inputs)
            
            # 손실 계산
            loss = criterion(outputs, labels)

            # 역전파 및 파라미터 업데이트
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 모델 저장 디렉터리 확인 및 생성
    save_dir = os.path.dirname(model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"'{save_dir}' 디렉터리를 생성했습니다.")

    torch.save(model.state_dict(), model_save_path)
    
    print("--- 모델 학습 완료 ---")
    print(f"학습된 모델이 '{model_save_path}'에 저장되었습니다.")

# validation 추가 버전
def train_model_v2(model: nn.Module,
                train_loader: DataLoader, 
                val_loader: DataLoader, # val_loader 추가
                criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                num_epochs: int,
                model_save_path: str): 
    """
    주어진 모델과 데이터로 학습을 수행하고, 검증 세트를 기준으로 최적의 모델을 저장합니다.

    Args:
        model (nn.Module): 학습할 PyTorch 모델.
        train_loader (DataLoader): 학습 데이터셋을 위한 DataLoader.
        val_loader (DataLoader): 검증 데이터셋을 위한 DataLoader.
        criterion (nn.Module): 손실 함수.
        optimizer (torch.optim.Optimizer): 옵티마이저.
        num_epochs (int): 총 학습 에포크 수.
        model_save_path (str): 최적의 모델을 저장할 경로.
    """
    print("--- 모델 학습 시작 ---")
    
    best_val_accuracy = 0.0  # 최고 검증 정확도 추적

    for epoch in range(num_epochs):
        model.train()  # 모델을 학습 모드로 설정
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        
        # --- 에포크마다 검증 수행 ---
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, is_training=True)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # 최고 정확도를 가진 모델 저장
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            
            # 모델 저장 디렉터리 확인 및 생성
            save_dir = os.path.dirname(model_save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"'{save_dir}' 디렉터리를 생성했습니다.")

            torch.save(model.state_dict(), model_save_path)
            print(f"  -> 최고 검증 정확도 갱신: {best_val_accuracy:.2f}%. 모델을 '{model_save_path}'에 저장했습니다.")

    print("--- 모델 학습 완료 ---")
    print(f"최종 저장된 모델의 검증 정확도: {best_val_accuracy:.2f}%")

def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, is_training: bool = False):
    """
    학습된 모델을 평가합니다.

    Args:
        model (nn.Module): 평가할 PyTorch 모델.
        test_loader (DataLoader): 테스트 데이터셋을 위한 DataLoader.
        criterion (nn.Module): 손실 함수.
        is_training (bool): 학습 루프 내에서 호출되었는지 여부. True이면 결과 튜플을 반환.
    """
    if not is_training:
        print("\n--- 모델 평가 시작 ---")
        
    model.eval()  # 모델을 평가 모드로 설정
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = (correct_predictions / total_samples) * 100
    
    if is_training:
        return avg_loss, accuracy

    print(f"평가 결과: 평균 손실 = {avg_loss:.4f}, 정확도 = {accuracy:.2f}%")
    print("--- 모델 평가 완료 ---")
