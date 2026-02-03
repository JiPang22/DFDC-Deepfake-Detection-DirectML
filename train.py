import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 데이터셋 클래스
class DeepfakeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터 분할
full_data = pd.read_csv("dataset_labels.csv")
train_df, val_df = train_test_split(full_data, test_size=0.2, random_state=42)
train_loader = DataLoader(DeepfakeDataset(train_df, transform), batch_size=32, shuffle=True)
val_loader = DataLoader(DeepfakeDataset(val_df, transform), batch_size=32, shuffle=False)

# 모델
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def train():
    print(f"🚀 학습/검증 시작! 디바이스: {device}")
    history = {'train_loss': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(1):  # 우선 1에폭만 진행
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 10배치마다 Val Acc 체크 (실시간 모니터링)
            if i % 10 == 0:
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    # 시간 관계상 Val의 일부(예: 10배치)만 샘플링하여 실시간 Acc 측정
                    for v_img, v_lbl in val_loader:
                        v_img, v_lbl = v_img.to(device), v_lbl.to(device)
                        v_out = model(v_img)
                        _, pred = torch.max(v_out.data, 1)
                        total += v_lbl.size(0)
                        correct += (pred == v_lbl).sum().item()
                        if total > 320: break  # 실시간성 유지 위해 320개만 검증

                acc = 100 * correct / total
                history['train_loss'].append(loss.item())
                history['val_acc'].append(acc)

                elapsed = time.time() - start_time
                eta = (len(train_loader) - (i + 1)) * (elapsed / (i + 1))
                print(
                    f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f} | Val Acc: {acc:.2f}% | ETA: {int(eta // 60)}m {int(eta % 60)}s")
                model.train()

    # 학습 종료 후 모델 저장
    torch.save(model.state_dict(), "resnet18_deepfake.pth")

    # 학습 곡선 윈도우 출력
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Acc', color='orange')
    plt.title('Validation Accuracy')
    plt.legend()

    print("📈 학습 곡선을 별도 창으로 띄웁니다.")
    plt.show()


if __name__ == "__main__":
    train()