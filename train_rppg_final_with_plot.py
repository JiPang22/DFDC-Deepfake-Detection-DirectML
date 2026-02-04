import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# RX 6600 호환성 설정
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'


# 1. 데이터셋 및 모델 정의 (이전과 동일)
class BiometricDataset(Dataset):
    def __init__(self, base_path, transform=None, seq_length=10):
        self.base_path = base_path
        self.transform = transform
        self.seq_length = seq_length
        self.videos = []
        for label in ['real', 'fake']:
            folder = os.path.join(base_path, label)
            target = 0 if label == 'real' else 1
            video_frames = {}
            for f in os.listdir(folder):
                if f.endswith('.jpg'):
                    v_id = f.split('_')[1]
                    if v_id not in video_frames: video_frames[v_id] = []
                    video_frames[v_id].append(os.path.join(folder, f))
            for v_id in video_frames:
                frames = sorted(video_frames[v_id])
                if len(frames) >= seq_length:
                    self.videos.append((frames[:seq_length], target))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frame_paths, label = self.videos[idx]
        frames = [self.transform(Image.open(p).convert('L')) for p in frame_paths]
        return torch.stack(frames), label


class RPPGDeepfakeModel(nn.Module):
    def __init__(self):
        super(RPPGDeepfakeModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(64 * 7 * 7, 128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        b, s, c, h, w = x.size()
        cnn_feats = [self.cnn(x[:, i, :, :, :]) for i in range(s)]
        combined = torch.stack(cnn_feats, dim=1)
        _, (hidden, _) = self.lstm(combined)
        return self.fc(hidden[-1])


# 2. 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
dataset = BiometricDataset("/home/jplinux/Deepfake_project/train_data_85", transform=transform)
train_size = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

model = RPPGDeepfakeModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()
history = {'train_acc': [], 'val_acc': []}

# 3. 학습 루프
print("rPPG + ViViT 구조 심화 학습 시작...")
try:
    for epoch in range(30):
        model.train()
        t_correct, t_total = 0, 0
        start_t = time.time()
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            t_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            t_total += labels.size(0)

        train_acc = t_correct / t_total
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                v_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
                v_total += labels.size(0)

        val_acc = v_correct / v_total
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 예상 종료 시간(ETA) 출력
        print(
            f"ETA: {(time.time() - start_t) * (30 - epoch - 1) / 60:.1f}분 | Acc: {train_acc:.4f}_ Val_Acc: {val_acc:.4f}_")

        # [주문 사항] 0.5% 오차 시 중단
        if abs(train_acc - val_acc) <= 0.005 and epoch > 2:
            print(f"★ 조기 종료 조건 충족. 모델 저장.")
            torch.save(model.state_dict(), "deepfake_rppg_best.pth")
            break
finally:
    # 4. 시각화 윈도우 팝업 (학습 중단 시에도 실행되도록 finally 처리)
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], marker='o', label='Train Acc')
    plt.plot(history['val_acc'], marker='x', label='Val Acc')
    plt.title('Training Result (rPPG + ViViT)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('rppg_accuracy_plot.png')
    print("\n[알림] 학습 곡선 그래프 창을 띄웁니다. 창을 닫아야 프로그램이 완전히 종료됩니다.")
    plt.show()  # 윈도우 창 띄우기