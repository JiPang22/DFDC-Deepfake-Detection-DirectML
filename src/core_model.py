import os
import sys
# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# [파일 경로: /home/jplinux/Deepfake_project/src/core_model.py]
import torch
import torch.nn as nn
import torch.nn.functional as F

class RPPGDeepfakeModel(nn.Module):
    def __init__(self):
        super(RPPGDeepfakeModel, self).__init__()
        
        # --- [Branch 1: Spatial (공간적 특징)] ---
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        self.fc_spatial = nn.Linear(4096, 128)

        # --- [Branch 2: Frequency (주파수 특징)] ---
        self.fc_freq = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        # --- [Fusion & Classifier] ---
        self.fc_final = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
            # [수정] Sigmoid 제거 (BCEWithLogitsLoss 사용을 위해 Logits 반환)
        )

    def forward(self, x):
        b, c, t, h, w = x.size()

        # 1. Spatial Branch
        x_spatial = x.contiguous().view(-1, c, h, w)
        x_spatial = self.pool(F.relu(self.bn1(self.conv1(x_spatial))))
        x_spatial = self.pool(F.relu(self.bn2(self.conv2(x_spatial))))
        x_spatial = self.adaptive_pool(x_spatial)
        x_spatial = torch.flatten(x_spatial, 1)
        x_spatial = F.relu(self.fc_spatial(x_spatial))
        x_spatial = x_spatial.view(b, t, -1)
        x_spatial = torch.mean(x_spatial, dim=1)

        # 2. Frequency Branch
        signal = torch.mean(x, dim=(3, 4)) 
        fft_complex = torch.fft.rfft(signal, dim=2) 
        fft_mag = torch.abs(fft_complex).squeeze(1)
        
        if fft_mag.size(1) != 16:
             fft_mag = F.interpolate(fft_mag.unsqueeze(1), size=16, mode='linear', align_corners=False).squeeze(1)

        x_freq = self.fc_freq(fft_mag)

        # 3. Fusion
        combined = torch.cat((x_spatial, x_freq), dim=1)
        output = self.fc_final(combined)
        
        return output # Logits 반환

if __name__ == "__main__":
    model = RPPGDeepfakeModel()
    test_input = torch.randn(4, 1, 30, 128, 128)
    output = model(test_input)
    print(f"모델 테스트 완료. 출력 형상: {output.shape}")
