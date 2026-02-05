import os
import shutil
import glob

def add_path_header(file_path):
    """
    모든 파이썬 스크립트 상단에 프로젝트 루트를 sys.path에 추가하는 코드를 삽입합니다.
    이게 있어야 폴더가 바뀌어도 import 에러가 안 납니다.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 이미 헤더가 있는지 확인
    if "project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))" in "".join(lines[:20]):
        return

    # 헤더 작성 (상위 폴더를 path에 추가)
    header = [
        "import os\n",
        "import sys\n",
        "# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정\n",
        "project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n",
        "if project_root not in sys.path:\n",
        "    sys.path.append(project_root)\n",
        "\n"
    ]

    # 기존 import os, sys 제거 (중복 방지)
    new_lines = []
    for line in lines:
        if line.strip() in ["import os", "import sys"]:
            continue
        new_lines.append(line)

    # 파일 다시 쓰기
    with open(file_path, 'w') as f:
        f.writelines(header + new_lines)
    print(f"[Patch] 경로 헤더 추가 완료: {os.path.basename(file_path)}")

def fix_imports(file_path):
    """
    상대 경로 문제 해결:
    from core_model -> from src.core_model
    from dataset -> from src.dataset
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # 단순 치환 (가장 확실한 방법)
    replacements = {
        "from core_model": "from src.core_model",
        "from dataset": "from src.dataset",
        "from evaluate": "from scripts.evaluate",
        "from inference": "from scripts.inference",
        "from visualize_results": "from scripts.visualize_results",
        # 데이터 경로 수정
        "'/home/jplinux/Deepfake_project/train_data_85'": "os.path.join(project_root, 'data/raw/train_data_85')",
        "'/home/jplinux/Deepfake_project/data/processed'": "os.path.join(project_root, 'data/processed')",
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    # 중복 수정 방지 (src.src.core_model 같은 경우)
    content = content.replace("src.src.", "src.")
    content = content.replace("scripts.scripts.", "scripts.")

    with open(file_path, 'w') as f:
        f.write(content)
    print(f"[Fix] 임포트 구문 수정 완료: {os.path.basename(file_path)}")

def finalize():
    root_dir = '/home/jplinux/Deepfake_project'
    src_dir = os.path.join(root_dir, 'src')
    scripts_dir = os.path.join(root_dir, 'scripts')
    data_raw_dir = os.path.join(root_dir, 'data', 'raw')

    # 1. 폴더 생성
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(data_raw_dir, exist_ok=True)
    
    # __init__.py 생성 (패키지 인식용)
    open(os.path.join(src_dir, '__init__.py'), 'a').close()
    open(os.path.join(scripts_dir, '__init__.py'), 'a').close()

    # 2. 파일 이동
    moves = {
        'preprocess_from_images.py': scripts_dir,
        'visualize_results.py': scripts_dir,
        'evaluate.py': scripts_dir,
        'inference.py': scripts_dir,
        'check_files.py': scripts_dir,
        'cleanup_project.py': scripts_dir
    }

    for filename, dest in moves.items():
        old_path = os.path.join(src_dir, filename)
        if os.path.exists(old_path):
            shutil.move(old_path, os.path.join(dest, filename))
            print(f"[Move] {filename} -> scripts/")

    # 데이터 이동
    old_data = os.path.join(root_dir, 'train_data_85')
    if os.path.exists(old_data):
        shutil.move(old_data, os.path.join(data_raw_dir, 'train_data_85'))
        print(f"[Move] train_data_85 -> data/raw/")

    # 3. 코드 패치 (헤더 추가 + 임포트 수정)
    all_files = glob.glob(os.path.join(src_dir, '*.py')) + \
                glob.glob(os.path.join(scripts_dir, '*.py'))

    for py_file in all_files:
        if "__init__" in py_file: continue
        add_path_header(py_file)
        fix_imports(py_file)

    print("\n✅ 프로젝트 구조 최적화 및 코드 보정이 완료되었습니다.")
    print("이제 'python src/train_final.py' 처럼 루트에서 실행하시면 됩니다.")

if __name__ == "__main__":
    finalize()