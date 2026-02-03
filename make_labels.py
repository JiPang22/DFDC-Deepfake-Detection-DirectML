import os
import pandas as pd

# 얼굴 데이터가 저장된 경로
face_dir = "/home/jplinux/Deepfake_project/processed_faces"
output_csv = "dataset_labels.csv"


def generate_labels():
    data = []
    files = [f for f in os.listdir(face_dir) if f.endswith('.jpg')]

    for file in files:
        # 파일명 규칙에 따라 라벨링 (파일명에 'fake'가 있거나 특정 규칙이 있다면 수정 필요)
        # 우선 테스트를 위해 모든 사진을 임시로 'Fake(1)'로 분류하거나
        # 실제 DFDC 메타데이터(json)를 읽어와야 하지만,
        # 지금은 코드 작동 확인을 위해 0과 1을 섞어서 리스트를 만듭니다.

        label = 1 if "fake" in file.lower() else 0
        data.append({"path": os.path.join(face_dir, file), "label": label})

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ 총 {len(df)}개의 데이터 라벨링 완료! -> {output_csv}")


if __name__ == "__main__":
    generate_labels()