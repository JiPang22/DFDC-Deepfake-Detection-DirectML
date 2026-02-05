import os
import sys
# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# [파일 경로: /home/jplinux/Deepfake_project/src/train_final.py]
import time
import datetime # [추가] 시간 계산용
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import torchvision.transforms.functional as TF
import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.core_model import RPPGDeepfakeModel
from src.dataset import DeepfakeDataset
from scripts.evaluate import evaluate
from scripts.inference import run_inference_random
from scripts.visualize_results import visualize

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

# [색상 코드 정의]
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

logging.basicConfig(
    filename=os.path.join(current_dir, 'training_final_log.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# [정밀 가공 1] Focal Loss
class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def add_advanced_augmentation(tensor):
    """[데이터 증강: 특전사 모드]"""
    if torch.rand(1).item() < 0.5:
        tensor = torch.flip(tensor, dims=[4])
    if torch.rand(1).item() < 0.5:
        factor = torch.empty(1).uniform_(0.7, 1.3).item()
        tensor = tensor * factor
    if torch.rand(1).item() < 0.5:
        factor = torch.empty(1).uniform_(0.8, 1.2).item()
        mean = tensor.mean()
        tensor = (tensor - mean) * factor + mean
    if torch.rand(1).item() < 0.5:
        noise = torch.randn_like(tensor) * 0.03
        tensor = tensor + noise
    if torch.rand(1).item() < 0.3:
        b, c, t, h, w = tensor.shape
        tensor_reshaped = tensor.view(-1, c, h, w)
        sigma = torch.empty(1).uniform_(0.1, 2.0).item()
        tensor_reshaped = TF.gaussian_blur(tensor_reshaped, kernel_size=3, sigma=[sigma, sigma])
        tensor = tensor_reshaped.view(b, c, t, h, w)
    return torch.clamp(tensor, 0.0, 1.0)

def calculate_accuracy(outputs, labels):
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    correct = (predicted == labels).float().sum()
    return correct / labels.size(0)

def train_final():
    # --- [최종 학습 설정] ---
    DATA_ROOT = os.path.join(project_root, 'data/processed')
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 4
    EPOCHS = 200 
    LR = 0.001
    PATIENCE = 20 
    TARGET_ACC = 0.80 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{Colors.GREEN}[System] Device: {device}{Colors.RESET}")
    print(f"{Colors.GREEN}[Info] Epochs: {EPOCHS} (Focal Loss + Cosine Annealing){Colors.RESET}")
    print(f"{Colors.GREEN}[Control] Close graph window to stop & save.{Colors.RESET}")
    logging.info(f"Final Training Started on {device}")

    # 1. 시각화 초기화
    plot_enabled = True
    stop_training = False

    def on_close(event):
        nonlocal stop_training
        stop_training = True
        print(f"\n{Colors.GREEN}[Control] Graph window closed. Stopping...{Colors.RESET}")

    try:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.canvas.manager.set_window_title('Final Training Monitor (Initializing...)')
        fig.canvas.mpl_connect('close_event', on_close)
        
        ax1.set_title('Training Loss (Waiting for data...)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        ax2.set_title('Accuracy (Waiting for data...)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.pause(0.1)
        
    except Exception as e:
        print(f"{Colors.RED}[Warning] Plot init failed: {e}{Colors.RESET}")
        plot_enabled = False

    # 2. 데이터셋 로드
    if stop_training: return

    print(f"{Colors.GREEN}[Data] Loading dataset...{Colors.RESET}")
    full_dataset = DeepfakeDataset(data_root=DATA_ROOT)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 3. 모델 & 최적화
    model = RPPGDeepfakeModel().to(device)
    criterion = BinaryFocalLossWithLogits(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    history = {'loss': [], 'acc': [], 'val_acc': []}
    max_val_acc = 0.0
    patience_counter = 0
    
    print(f"{Colors.GREEN}Training Started...{Colors.RESET}")
    start_time = time.time()
    total_steps = EPOCHS * len(train_loader) # 전체 스텝 수 계산
    
    try:
        for epoch in range(EPOCHS):
            if stop_training: break

            model.train()
            running_loss = 0.0
            running_acc = 0.0
            
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
            
            for i, (inputs, labels) in enumerate(pbar):
                if stop_training: break

                inputs, labels = inputs.to(device), labels.to(device)
                inputs = add_advanced_augmentation(inputs)
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss = loss / ACCUMULATION_STEPS
                
                scaler.scale(loss).backward()
                
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                current_loss = loss.detach().item() * ACCUMULATION_STEPS
                current_acc = calculate_accuracy(outputs.detach(), labels).item()
                
                running_loss += current_loss
                running_acc += current_acc
                
                pbar.set_postfix({'Loss': f"{current_loss:.4f}", 'Acc': f"{current_acc:.4f}"})
                
                # [ETA 계산 및 그래프 업데이트]
                if plot_enabled:
                    try:
                        elapsed = time.time() - start_time
                        current_step = epoch * len(train_loader) + i + 1
                        if current_step > 0:
                            avg_time = elapsed / current_step
                            remaining_steps = total_steps - current_step
                            eta_seconds = int(avg_time * remaining_steps)
                            eta_str = str(datetime.timedelta(seconds=eta_seconds))
                            
                            # 그래프 제목에 ETA 표시
                            fig.suptitle(f'Final Training Monitor | ETA: {eta_str}', fontsize=12, color='blue')
                            fig.canvas.flush_events()
                    except: pass

            if stop_training: break

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = running_acc / len(train_loader)
            
            model.eval()
            val_acc = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with autocast():
                        outputs = model(inputs)
                    val_acc += calculate_accuracy(outputs, labels).item()
            
            epoch_val_acc = val_acc / len(val_loader)
            
            history['loss'].append(epoch_loss)
            history['acc'].append(epoch_acc)
            history['val_acc'].append(epoch_val_acc)
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            log_msg = f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {epoch_val_acc:.4f} | LR: {current_lr:.6f}"
            print(f"{Colors.GREEN}{log_msg}{Colors.RESET}")
            logging.info(log_msg)
            
            if plot_enabled:
                try:
                    ax1.clear()
                    ax1.plot(history['loss'], 'r-', label='Loss (Focal)')
                    ax1.set_title('Training Loss')
                    ax1.grid(True)
                    
                    ax2.clear()
                    ax2.plot(history['acc'], 'b-', label='Train Acc')
                    ax2.plot(history['val_acc'], 'g--', label='Val Acc')
                    ax2.set_title('Accuracy')
                    ax2.legend()
                    ax2.grid(True)
                    plt.tight_layout()
                    fig.canvas.flush_events()
                except Exception:
                    plot_enabled = False

            if epoch_val_acc >= TARGET_ACC and max_val_acc < TARGET_ACC:
                print(f"{Colors.GREEN}\n[Info] Target Accuracy {TARGET_ACC*100}% Reached: {epoch_val_acc*100:.2f}%{Colors.RESET}")
                logging.info(f"TARGET ACCURACY REACHED: {epoch_val_acc}")

            if epoch_val_acc > max_val_acc:
                max_val_acc = epoch_val_acc
                patience_counter = 0
                save_path = os.path.join(current_dir, 'rppg_best_real.pth')
                torch.save(model.state_dict(), save_path)
                print(f"{Colors.GREEN}  [Save] Model checkpoint saved: {epoch_val_acc:.4f}{Colors.RESET}")
                logging.info(f"Model Saved: {epoch_val_acc:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                print(f"{Colors.GREEN}\n[Stop] Early stopping triggered.{Colors.RESET}")
                break

    except KeyboardInterrupt:
        print(f"{Colors.GREEN}\n[Stop] Training interrupted by user.{Colors.RESET}")
        
    finally:
        if plot_enabled:
            plt.ioff()
            plt.close()
            
        total_time = (time.time() - start_time) / 3600
        print(f"{Colors.GREEN}\n[Done] Total Time: {total_time:.2f} hours{Colors.RESET}")
        print(f"{Colors.GREEN}Max Val Acc: {max_val_acc*100:.2f}%{Colors.RESET}")
        
        print("\n" + "="*50)
        print(f"{Colors.GREEN}🚀 Starting Automated Pipeline...{Colors.RESET}")
        print("="*50)
        
        try: evaluate()
        except Exception as e: print(f"{Colors.RED}[Error] Eval: {e}{Colors.RESET}")

        try: run_inference_random()
        except Exception as e: print(f"{Colors.RED}[Error] Infer: {e}{Colors.RESET}")

        try: visualize()
        except Exception as e: print(f"{Colors.RED}[Error] Viz: {e}{Colors.RESET}")
            
        print(f"{Colors.GREEN}\n✅ All tasks completed.{Colors.RESET}")

if __name__ == '__main__':
    train_final()