import os
import sys
# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# [파일 경로: /home/jplinux/Deepfake_project/src/train.py]
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
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

# [파이프라인 모듈]
from scripts.evaluate import evaluate
from scripts.inference import run_inference_random
from scripts.visualize_results import visualize

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

logging.basicConfig(
    filename=os.path.join(current_dir, 'training_log.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def calculate_accuracy(outputs, labels):
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    correct = (predicted == labels).float().sum()
    return correct / labels.size(0)

def train():
    # --- [기본 학습 설정] ---
    DATA_ROOT = os.path.join(project_root, 'data/processed')
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 4
    EPOCHS = 30
    LR = 0.001
    PATIENCE = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Device: {device}")
    print(f"[Info] Epochs: {EPOCHS}")
    print(f"[Control] Close graph window to stop & save.")
    logging.info(f"Standard Training Started on {device}")

    full_dataset = DeepfakeDataset(data_root=DATA_ROOT)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = RPPGDeepfakeModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    plot_enabled = True
    stop_training = False

    def on_close(event):
        nonlocal stop_training
        stop_training = True
        print("\n[Control] Graph window closed. Stopping...")

    try:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.canvas.manager.set_window_title('Standard Training Monitor')
        fig.canvas.mpl_connect('close_event', on_close)
    except Exception as e:
        print(f"[Warning] Plot init failed: {e}")
        plot_enabled = False

    history = {'loss': [], 'acc': [], 'val_acc': []}
    max_val_acc = 0.0
    patience_counter = 0
    
    print(f"Training Started...")
    start_time = time.time()
    
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
                
                if plot_enabled:
                    try: fig.canvas.flush_events()
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
            
            scheduler.step(epoch_val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            log_msg = f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {epoch_val_acc:.4f} | LR: {current_lr:.6f}"
            print(log_msg)
            logging.info(log_msg)
            
            if plot_enabled:
                try:
                    ax1.clear()
                    ax1.plot(history['loss'], 'r-', label='Loss')
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

            if epoch_val_acc > max_val_acc:
                max_val_acc = epoch_val_acc
                patience_counter = 0
                save_path = os.path.join(current_dir, 'rppg_best_real.pth')
                torch.save(model.state_dict(), save_path)
                print(f"  [Save] Model checkpoint saved: {epoch_val_acc:.4f}")
                logging.info(f"Model Saved: {epoch_val_acc:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                print(f"\n[Stop] Early stopping triggered.")
                break

    except KeyboardInterrupt:
        print("\n[Stop] Training interrupted by user.")
        
    finally:
        if plot_enabled:
            plt.ioff()
            plt.close()
            
        total_time = (time.time() - start_time) / 60
        print(f"\n[Done] Total Time: {total_time:.1f} min")
        print(f"Max Val Acc: {max_val_acc*100:.2f}%")
        
        print("\n" + "="*50)
        print("🚀 Starting Automated Pipeline...")
        print("="*50)
        
        try: evaluate()
        except Exception as e: print(f"[Error] Eval: {e}")

        try: run_inference_random()
        except Exception as e: print(f"[Error] Infer: {e}")

        try: visualize()
        except Exception as e: print(f"[Error] Viz: {e}")
            
        print("\n✅ All tasks completed.")

if __name__ == '__main__':
    train()