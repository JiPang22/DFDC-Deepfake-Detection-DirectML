# [파일 경로: /home/jplinux/Deepfake_project/src/train_warmup.py]
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import traceback

# [중요] 파이참 내부 뷰어 대신 별도 윈도우(Tkinter) 사용 강제
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass 

import matplotlib.pyplot as plt

# [경로 자동 설정]
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from core_model import RPPGDeepfakeModel

# [환경 변수 설정]
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

def add_noise(tensor, noise_factor=0.05):
    noise = torch.randn_like(tensor) * noise_factor
    return tensor + noise

def calculate_accuracy(outputs, labels):
    predicted = (outputs > 0.5).float()
    correct = (predicted == labels).float().sum()
    return correct / labels.size(0)

def train_step():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] 학습 장치 설정: {device}")

    model = RPPGDeepfakeModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    real_batch_size = 8
    accumulation_steps = 4
    epochs = 100 
    total_batches = 10 
    
    train_inputs = torch.randn(real_batch_size, 1, 30, 128, 128).to(device)
    train_labels = torch.randint(0, 2, (real_batch_size, 1)).float().to(device)
    
    val_inputs = torch.randn(real_batch_size, 1, 30, 128, 128).to(device)
    val_labels = torch.randint(0, 2, (real_batch_size, 1)).float().to(device)

    history = {'loss': [], 'acc': [], 'val_acc': []}
    
    # --- [실시간 시각화 설정] ---
    plt.ion() 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title('Real-time Training Monitor')
    fig.suptitle('Real-time Learning Curve')

    print(f"학습 시작: Max {epochs} 에폭")
    #print("종료 조건: Loss < 0.001 OR |Acc - Val_Acc| > 1% (과적합 감지)")

    start_time = time.time()
    model.train() 

    for epoch in range(epochs):
        epoch_loss, epoch_acc = 0.0, 0.0
        optimizer.zero_grad()
        
        for i in range(total_batches):
            outputs = model(add_noise(train_inputs))
            loss = criterion(outputs, train_labels)
            acc = calculate_accuracy(outputs, train_labels)
            
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            epoch_acc += acc.item()

        avg_loss = epoch_loss / total_batches
        avg_acc = epoch_acc / total_batches
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_inputs)
            val_acc = calculate_accuracy(val_outputs, val_labels).item()
        model.train()

        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        history['val_acc'].append(val_acc)

        # --- [실시간 그래프 업데이트] ---
        ax1.clear()
        ax1.plot(history['loss'], label='Train Loss', color='red')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        ax2.clear()
        ax2.plot(history['acc'], label='Train Acc', color='blue')
        ax2.plot(history['val_acc'], label='Val Acc', color='green', linestyle='--')
        ax2.set_title('Accuracy (Train vs Val)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.canvas.flush_events() 

        sample_pred = outputs[0].item()
        sample_label = train_labels[0].item()
        print(f"Epoch [{epoch+1}/{epochs}] L: {avg_loss:.4f} | A: {avg_acc:.4f} | V: {val_acc:.4f} | S(P/GT): {sample_pred:.4f}/{sample_label:.0f}")

        # --- [조기 종료 조건 수정] ---
        acc_diff = abs(avg_acc - val_acc)
        
        # 1. 성공 조건: Loss가 충분히 떨어짐
        if avg_loss < 0.001:
            print(f"\n[성공 종료] Loss가 0에 수렴했습니다. (Loss: {avg_loss:.6f})")
            break
            
        # 2. 실패 방지 조건: Acc 차이가 1% 이상 벌어짐 (과적합)
        # 초기 5에폭은 학습 안정화 기간으로 간주하여 제외
        Criterion=0.00001 #과적합 판단 기준
        if epoch > 5 and acc_diff > Criterion:
            print(f"\n[경고 종료] Train/Val 정확도 격차가 {Criterion*100:.0f}%를 초과했습니다. (Diff: {acc_diff:.4f})")
            print("해석: 모델이 훈련 데이터에 과적합되기 시작했거나, 일반화에 실패했습니다.")
            break

    save_path = os.path.join(current_dir, "../src/rppg_best_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"모델 저장 완료: {save_path}")
    
    plt.ioff() 
    print("학습 종료. 그래프 창을 닫으면 프로그램이 종료됩니다.")
    plt.show()

if __name__ == "__main__":
    train_step()