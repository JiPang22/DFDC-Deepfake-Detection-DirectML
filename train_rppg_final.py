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


# 1. rPPG 분석을 위한 시퀀스 데이터셋 (10프레임씩 묶음)
class BiometricDataset(Dataset):
    def __init__(self, base_path, transform=None, seq_length=10):
        self.base_path = base_path
        self.transform = transform
        self.seq_length = seq_length
        self.videos = []

        for label in ['real', 'fake']:
            folder = os.path.join(base_path, label)
            target = 0 if label == 'real' else 1
            # 파일명 규칙(label_videoIdx_frameIdx.jpg)을 이용해 영상 단위로 그룹화
            video_frames = {}
            for f in os.listdir(folder):
                if f.endswith('.jpg'):
                    parts = f.split('_')
                    v_id = parts[1]
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
        frames = []
        for p in frame_paths:
            img = Image.open(p).convert('L')
            if self.transform: img = self.transform(img)
            frames.append(img)
        return torch.stack(frames), label  # (Seq, C, H, W)


# 2. CNN + LSTM (rPPG 생체 신호 탐지 모델)
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
        batch_size, seq_len, c, h, w = x.size()
        cnn_feats = []
        for i in range(seq_len):
            cnn_feats.append(self.cnn(x[:, i, :, :, :]))
        combined = torch.stack(cnn_feats, dim=1)
        _, (hidden, _) = self.lstm(combined)
        return self.fc(hidden[-1])


# 3. 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])  # VRAM 절약을 위해 128 조정
dataset = BiometricDataset("/home/jplinux/Deepfake_project/train_data_85", transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  # 시퀀스 데이터는 메모리를 많이 먹으므로 배치 축소
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

model = RPPGDeepfakeModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

history = {'train_acc': [], 'val_acc': []}

# 4. 학습 루프 (ETA 포함)
print(f"rPPG 심화 학습 시작: {len(dataset)}개 시퀀스 탐지 중...")
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

        _, pred = torch.max(outputs, 1)
        t_total += labels.size(0)
        t_correct += (pred == labels).sum().item()

    train_acc = t_correct / t_total

    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            v_total += labels.size(0)
            v_correct += (pred == labels).sum().item()

    val_acc = v_correct / v_total
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(
        f"ETA: {(time.time() - start_t) * (30 - epoch - 1) / 60:.1f}분 남음 | Acc: {train_acc:.4f}_ Val_Acc: {val_acc:.4f}_")

    # [주문 사항] 0.5% 이내 오차 시 조기 종료 및 저장
    if abs(train_acc - val_acc) <= 0.005 and epoch > 2:
        print(f"★ 조기 종료 조건 충족(오차: {abs(train_acc - val_acc):.5f}). 모델을 저장합니다.")
        torch.save(model.state_dict(), "deepfake_rppg_best.pth")
        break

# 5. 시각화 저장
plt.figure(figsize=(10, 5))
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('rPPG Model Accuracy')
plt.legend()
plt.savefig('rppg_accuracy_plot.png')
print("학습 완료 및 그래프 저장됨.")