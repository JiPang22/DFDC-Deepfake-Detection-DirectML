import os
import shutil
import glob

def organize_project():
    """
    프로젝트 구조를 'src', 'scripts', 'data' 중심으로 재구성합니다.
    """
    root_dir = '/home/jplinux/Deepfake_project'
    
    # --- [1. 폴더 생성] ---
    scripts_dir = os.path.join(root_dir, 'scripts')
    data_raw_dir = os.path.join(root_dir, 'data', 'raw')
    
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(data_raw_dir, exist_ok=True)
    
    print("--- [1] 폴더 구조 재구성 시작 ---")

    # --- [2. 파일 이동] ---
    # 보조 스크립트 이동
    aux_scripts = [
        'preprocess_from_images.py',
        'visualize_results.py',
        'evaluate.py',
        'inference.py',
        'check_files.py',
        'cleanup_project.py'
    ]
    for script in aux_scripts:
        src = os.path.join(root_dir, 'src', script)
        dst = os.path.join(scripts_dir, script)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"[Move] {script} -> scripts/")

    # 원본 데이터 이동
    train_data_src = os.path.join(root_dir, 'train_data_85')
    train_data_dst = os.path.join(data_raw_dir, 'train_data_85')
    if os.path.exists(train_data_src) and not os.path.exists(train_data_dst):
        shutil.move(train_data_src, train_data_dst)
        print(f"[Move] train_data_85 -> data/raw/")

    print("\n--- [2. 스크립트 내부 경로 수정] ---")

    # --- [3. 경로 수정] ---
    files_to_update = glob.glob(os.path.join(root_dir, 'src', '*.py')) + \
                      glob.glob(os.path.join(scripts_dir, '*.py'))

    path_mappings = {
        # 이전 경로 -> 새 경로
        "'/home/jplinux/Deepfake_project/train_data_85'": "'/home/jplinux/Deepfake_project/data/raw/train_data_85'",
        "'/home/jplinux/Deepfake_project/data/processed'": "'/home/jplinux/Deepfake_project/data/processed'",
        # 파이프라인 임포트 경로 수정
        "from evaluate import evaluate": "from scripts.evaluate import evaluate",
        "from inference import run_inference_random": "from scripts.inference import run_inference_random",
        "from visualize_results import visualize": "from scripts.visualize_results import visualize",
    }

    for file_path in files_to_update:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            for old_path, new_path in path_mappings.items():
                content = content.replace(old_path, new_path)
            
            # src 폴더 내에서 다른 src 파일을 임포트하는 경우 (상대 경로로 변경)
            content = content.replace("from core_model", "from .core_model")
            content = content.replace("from dataset", "from .dataset")

            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"[Update] 경로 수정 완료: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"[Error] {file_path} 수정 실패: {e}")

    print("\n--- [3. 최종 구조] ---")
    print("- src/ (핵심 로직)")
    print("  - core_model.py")
    print("  - dataset.py")
    print("  - train*.py")
    print("- scripts/ (보조 스크립트)")
    print("  - preprocess_from_images.py")
    print("  - evaluate.py")
    print("  - ...")
    print("- data/")
    print("  - raw/ (원본 데이터)")
    print("  - processed/ (전처리된 데이터)")
    print("\n프로젝트 정리가 완료되었습니다.")

if __name__ == "__main__":
    organize_project()