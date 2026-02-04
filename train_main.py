import os
import sys
import time
from datetime import datetime, timedelta

# 1. RX 6600(gfx1032) 호환성 강제 해결
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# 2. 데이터셋 클래스 정의 (데이터셋 리스트 구성(라벨링))
class DeepfakeDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.base_path = base_path
        self.transform = transform
        self.samples = []
        for label in ['real', 'fake']:
            folder = os.path.join(base_path, label)
            target = 0 if label == 'real' else 1
            for f in os.listdir(folder):
                if f.endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(folder, f), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')  # 흑백(1채널) 변환으로 VRAM 절약
        if self.transform:
            img = self.transform(img)
        return img, label


# 3. 모델 설계 (ViViT 기반 경량화)
class LightweightViViT(nn.Module):
    def __init__(self):
        super(LightweightViViT, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 2)
        )

    def forward(self, x):
        return self.features(x)


# 4. 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = DeepfakeDataset("/home/jplinux/Deepfake_project/train_data_85", transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

model = LightweightViViT().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs = 10

# 5. 학습 루프
print(f"학습 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # ETA 계산
    epoch_duration = time.time() - start_time
    eta_seconds = (epochs - (epoch + 1)) * epoch_duration
    eta_datetime = datetime.now() + timedelta(seconds=eta_seconds)

    print(f"\n[Epoch {epoch + 1}] 평균 Loss: {running_loss / len(train_loader):.4f}")
    print(f"학습 종료 예상(ETA): {eta_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 30)

torch.save(model.state_dict(), "deepfake_model_final.pth")
print("최종 모델 저장 완료!")