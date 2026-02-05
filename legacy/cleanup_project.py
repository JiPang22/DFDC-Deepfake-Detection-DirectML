import os
import shutil

def cleanup():
    root_dir = '/'
    legacy_dir = os.path.join(root_dir, '')
    
    # 1. Legacy 폴더 생성
    if not os.path.exists(legacy_dir):
        os.makedirs(legacy_dir)
        print(f"[생성] {legacy_dir}")

    # 2. 이동할 파일 및 폴더 목록 정의
    # (현재 사용 중인 src, data, train_data_85, .venv, .git 등은 제외)
    items_to_move = [
        # Directories
        'oldThings',
        'models',
        'results',
        '__pycache__',
        
        # Python Scripts (Root level)
        'create_dataset.py',
        'data_loader.py',
        'extract_frames_fast.py',
        'imgtest.py',
        'make_labels.py',
        'oneToOneImg.py',
        'predict.py',
        'train_rppg_final.py',
        'uniJson.py',
        
        # CSV Files
        'balanced_video_list.csv',
        'dataset_labels.csv',
        'train_list.csv',
        
        # Old Model Weights
        'deepfake_model_final.pth',
        'resnet18_deepfake.pth'
    ]

    print("--- 정리 시작 ---")
    
    for item in items_to_move:
        src_path = os.path.join(root_dir, item)
        dst_path = os.path.join(legacy_dir, item)
        
        if os.path.exists(src_path):
            try:
                shutil.move(src_path, dst_path)
                print(f"[이동] {item} -> legacy/")
            except Exception as e:
                print(f"[오류] {item} 이동 실패: {e}")
        else:
            print(f"[스킵] {item} (파일 없음)")

    print("--- 정리 완료 ---")
    print(f"현재 루트에는 src/, data/, train_data_85/ 등 핵심 폴더만 남았습니다.")

if __name__ == "__main__":
    cleanup()