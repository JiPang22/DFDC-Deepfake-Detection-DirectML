import os
import sys
# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# [파일 경로: /home/jplinux/Deepfake_project/src/dataset.py]
import torch
from torch.utils.data import Dataset
import numpy as np

class DeepfakeDataset(Dataset):
    """
    실제 데이터셋을 관리하고 모델에 공급하는 클래스.
    - __init__: 데이터셋의 경로를 받아 파일 리스트와 라벨을 생성.
    - __len__: 전체 데이터의 개수를 반환.
    - __getitem__: 특정 인덱스의 데이터(텐서)와 라벨을 반환.
    """
    def __init__(self, data_root, transform=None):
        """
        데이터셋 초기화.
        - data_root: 데이터셋의 루트 폴더. e.g., '.../data/'
                     내부에 'real'과 'fake' 폴더가 있어야 함.
        - transform: 데이터에 적용할 전처리 (e.g., Augmentation)
        """
        self.data_root = data_root
        self.transform = transform
        self.file_list = []
        self.labels = []

        print(f"[Dataset] 데이터셋 스캔 중... 경로: {data_root}")

        # 'real' 데이터 (label=0)
        real_path = os.path.join(data_root, 'real')
        if os.path.exists(real_path):
            for file_name in os.listdir(real_path):
                self.file_list.append(os.path.join(real_path, file_name))
                self.labels.append(0)
        
        # 'fake' 데이터 (label=1)
        fake_path = os.path.join(data_root, 'fake')
        if os.path.exists(fake_path):
            for file_name in os.listdir(fake_path):
                self.file_list.append(os.path.join(fake_path, file_name))
                self.labels.append(1)
        
        if not self.file_list:
            print(f"[경고] '{data_root}' 경로에서 데이터를 찾을 수 없습니다.")
            print("         폴더 구조: data_root -> real/ , fake/")
        else:
            print(f"[Dataset] 스캔 완료. Real: {self.labels.count(0)}개, Fake: {self.labels.count(1)}개")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        데이터 로딩 및 전처리.
        - 현재는 전처리된 .npy 파일을 로드하는 것을 가정.
        - 추후 비디오 직접 로딩 및 프레임 추출 로직으로 확장 가능.
        """
        file_path = self.file_list[idx]
        label = self.labels[idx]

        # [가정] 데이터는 (C, T, H, W) 형태의 numpy 배열로 저장되어 있음
        # 예: (1, 30, 128, 128)
        try:
            # .npy 파일 로드
            data = np.load(file_path)
        except Exception as e:
            print(f"[오류] 파일 로드 실패: {file_path}. 에러: {e}")
            # 오류 발생 시 더미 데이터 반환
            data = np.zeros((1, 30, 128, 128), dtype=np.float32)
            label = -1 # 오류 라벨

        # Numpy to Tensor
        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.tensor([label], dtype=torch.float)

        # Transform 적용 (필요시)
        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label_tensor

# [단위 테스트]
if __name__ == '__main__':
    # [사용 예시]
    # 1. 가상 데이터셋 폴더 및 파일 생성
    print("[단위 테스트] 가상 데이터셋 생성 중...")
    test_data_dir = './temp_dataset'
    os.makedirs(os.path.join(test_data_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(test_data_dir, 'fake'), exist_ok=True)
    
    # 가짜 .npy 파일 생성
    sample_real = np.random.rand(1, 30, 128, 128).astype(np.float32)
    sample_fake = np.random.rand(1, 30, 128, 128).astype(np.float32)
    np.save(os.path.join(test_data_dir, 'real', 'vid1.npy'), sample_real)
    np.save(os.path.join(test_data_dir, 'fake', 'vid2.npy'), sample_fake)
    
    # 2. 데이터셋 인스턴스화 및 테스트
    dataset = DeepfakeDataset(data_root=test_data_dir)
    
    if len(dataset) > 0:
        print(f"\n데이터셋 길이: {len(dataset)}")
        first_data, first_label = dataset[0]
        print(f"첫 번째 데이터 형상: {first_data.shape}")
        print(f"첫 번째 라벨: {first_label.item()}")
        print("\n[성공] Dataset 클래스가 정상적으로 작동합니다.")
    else:
        print("\n[실패] Dataset 생성에 문제가 있습니다.")
        
    # 3. 임시 파일 정리
    import shutil
    shutil.rmtree(test_data_dir)
    print("\n[정리] 가상 데이터셋 삭제 완료.")