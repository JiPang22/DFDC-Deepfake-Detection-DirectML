import os
import sys
# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# [파일 경로: /home/jplinux/Deepfake_project/src/train_overnight.py]
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging

# [경로 설정]
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.core_model import RPPGDeepfakeModel
from src.dataset import DeepfakeDataset

# [환경 변수]
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

# [로깅 설정]
logging.basicConfig(
    filename=os.path.join(current_dir, 'training_log.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def add_augmentation(tensor):
    """
    [데이터 증강] 과적합 방지를 위한 실시간 변형
    1. Random Noise
    2. Random Horizontal Flip (Tensor 차원 조작)
    """
    # 1. Noise
    if torch.rand(1).item() < 0.5:
        noise = torch.randn_like(tensor) * 0.02
        tensor = tensor + noise
        
    # 2. Horizontal Flip (W 차원 뒤집기: index 4)
    # Tensor shape: (B, C, T, H, W)
    if torch.rand(1).item() < 0.5:
        tensor = torch.flip(tensor, dims=[4])
        
    return tensor

def calculate_accuracy(outputs, labels):
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    correct = (predicted == labels).float().sum()
    return correct / labels.size(0)

def train_overnight():
    # --- [설정: 밤샘 모드] ---
    DATA_ROOT = os.path.join(project_root, 'data/processed')
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 4
    EPOCHS = 200 # 넉넉하게 설정
    LR = 0.001
    PATIENCE = 15 # 성능 향상 없을 때 기다려줄 에폭 수
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] 학습 장치: {device}")
    logging.info(f"Training Started on {device}")

    # 1. 데이터셋
    full_dataset = DeepfakeDataset(data_root=DATA_ROOT)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 2. 모델 & 최적화
    model = RPPGDeepfakeModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4) # L2 Regularization 추가
    scaler = GradScaler()
    
    # [스케줄러] 성능 정체 시 학습률 감소 (Factor 0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"밤샘 학습 시작: 총 {EPOCHS} 에폭 (Scheduler + Augmentation)")
    start_time = time.time()
    
    try:
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
            
            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # [증강] 학습 데이터에만 적용
                inputs = add_augmentation(inputs)
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss = loss / ACCUMULATION_STEPS
                
                scaler.scale(loss).backward()
                
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                current_loss = loss.detach().item() * ACCUMULATION_STEPS
                current_acc = calculate_accuracy(outputs.detach(), labels).item()
                
                running_loss += current_loss
                running_acc += current_acc
                
                pbar.set_postfix({'Loss': f"{current_loss:.4f}", 'Acc': f"{current_acc:.4f}"})
                
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = running_acc / len(train_loader)
            
            # Validation
            model.eval()
            val_acc = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with autocast():
                        outputs = model(inputs)
                    val_acc += calculate_accuracy(outputs, labels).item()
            
            epoch_val_acc = val_acc / len(val_loader)
            
            # 스케줄러 업데이트
            scheduler.step(epoch_val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 로그 기록
            log_msg = f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {epoch_val_acc:.4f} | LR: {current_lr:.6f}"
            print(log_msg)
            logging.info(log_msg)
            
            # 모델 저장 및 Early Stopping 체크
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                patience_counter = 0
                save_path = os.path.join(current_dir, 'rppg_best_overnight.pth')
                torch.save(model.state_dict(), save_path)
                print(f"  >>> 최고 성능 갱신! 저장됨: {epoch_val_acc:.4f}")
                logging.info(f"New Best Model Saved: {epoch_val_acc:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                print(f"\n[조기 종료] {PATIENCE} 에폭 동안 성능 향상이 없어 학습을 종료합니다.")
                logging.info("Early Stopping Triggered")
                break

    except KeyboardInterrupt:
        print("\n[사용자 중단] 학습 종료 및 저장.")
        logging.info("Training Interrupted by User")
        
    except Exception as e:
        print(f"\n[오류 발생] {e}")
        logging.error(f"Error occurred: {e}")
        
    finally:
        # 마지막 상태 저장
        final_path = os.path.join(current_dir, 'rppg_last_checkpoint.pth')
        torch.save(model.state_dict(), final_path)
        
        total_time = (time.time() - start_time) / 3600
        print(f"학습 종료. 총 소요 시간: {total_time:.2f}시간")
        logging.info(f"Training Finished. Total Time: {total_time:.2f} hours")

if __name__ == '__main__':
    train_overnight()