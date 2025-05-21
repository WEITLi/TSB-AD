import os
import subprocess
import sys
from pathlib import Path

def setup_colab():
    """设置Colab环境"""
    # 检查是否在Colab环境中
    if 'google.colab' not in sys.modules:
        print("这不是Colab环境，请确保在Google Colab中运行此脚本")
        return

    # 挂载Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # 设置项目根目录
    PROJECT_ROOT = "/content/drive/MyDrive/TSB-AD/TSB-AD"
    os.chdir(PROJECT_ROOT)

    # 创建必要的目录
    directories = [
        'eval/benchmark',
        'eval/metrics/uni',
        'eval/metrics/multi',
        'eval/score/uni',
        'eval/score/multi'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # 设置环境变量
    os.environ['PROJECT_ROOT'] = PROJECT_ROOT
    os.environ['DATASET_DIR'] = os.path.join(PROJECT_ROOT, 'Datasets')

    # 检查conda是否已安装
    if not os.path.exists('/usr/local/bin/conda'):
        print("正在安装Miniconda...")
        subprocess.run(['wget', 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'], check=True)
        subprocess.run(['chmod', '+x', 'Miniconda3-latest-Linux-x86_64.sh'], check=True)
        subprocess.run(['bash', './Miniconda3-latest-Linux-x86_64.sh', '-b', '-f', '-p', '/usr/local'], check=True)
        print("Miniconda安装完成")

    # 检查环境是否存在
    env_exists = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True).stdout
    if 'tsb_env' not in env_exists:
        print("正在创建conda环境...")
        subprocess.run(['conda', 'env', 'create', '-f', 'environment.yml'], check=True)
        print("环境创建完成")
    else:
        print("环境已存在，正在更新...")
        subprocess.run(['conda', 'env', 'update', '-f', 'environment.yml'], check=True)
        print("环境更新完成")

    # 激活环境
    subprocess.run(['conda', 'run', '-n', 'tsb_env', 'python', '-c', 'import sys; print(sys.executable)'], check=True)

    print("\nColab环境初始化完成！")
    print("数据集目录:", os.environ['DATASET_DIR'])
    print("\n使用以下命令激活环境：")
    print("!conda activate tsb_env")

if __name__ == "__main__":
    setup_colab() 