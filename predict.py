import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# 1. GPU 호환성 설정
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. 모델 구조 (학습 시와 동일)
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


# 3. 모델 로드 (weights_only 설정으로 보안 경고 해결)
model = LightweightViViT().to(device)
model.load_state_dict(torch.load("deepfake_model_final.pth", weights_only=True))
model.eval()

# 4. 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


def predict(image_path):
    print(f"판별 대상: {image_path}")
    img = Image.open(image_path)
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = "REAL" if predicted.item() == 0 else "FAKE"
    print(f"결과: {label} (확률: {confidence.item() * 100:.2f}%)")


if __name__ == "__main__":
    # 폴더 내 파일 하나를 자동으로 선택
    target_folder = "/home/jplinux/Deepfake_project/train_data_85/fake"
    files = [f for f in os.listdir(target_folder) if f.endswith('.jpg')]

    if files:
        test_img = os.path.join(target_folder, files[0])
        predict(test_img)
    else:
        print("테스트할 이미지 파일이 폴더에 없습니다.")