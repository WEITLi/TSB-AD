import os
import subprocess
import sys
import json

def initialize_colab_environment():
    print("开始 Colab 环境初始化...")
    original_cwd = os.getcwd()
    print(f"脚本启动时的原始当前工作目录: {original_cwd}")

    # 1. 确定并切换到项目根目录
    #    Google Drive挂载应在此脚本运行前完成。
    #    确保此路径与你在Google Drive中的实际项目路径完全一致。
    EXPECTED_PROJECT_ROOT = "/content/TSB-AD/TSB-AD"
    ENV_FILE_NAME = "environment.yml"

    print(f"期望的项目根目录: {EXPECTED_PROJECT_ROOT}")

    if os.path.isdir(EXPECTED_PROJECT_ROOT):
        print(f"期望的项目根目录存在。正在尝试切换...")
        try:
            os.chdir(EXPECTED_PROJECT_ROOT)
            print(f"成功切换到项目目录: {os.getcwd()}")
        except Exception as e:
            print(f"错误: 尝试切换到 {EXPECTED_PROJECT_ROOT} 失败: {e}")
            print(f"请检查路径是否正确，以及Google Drive是否已正确挂载并可访问。")
            return
    else:
        print(f"警告: 期望的项目根目录 {EXPECTED_PROJECT_ROOT} 未找到或不是一个目录。")
        print(f"将尝试在当前工作目录 ({original_cwd}) 中查找 {ENV_FILE_NAME}。")
        # 如果期望的目录不存在，我们就不改变目录，假设用户已在正确的目录运行脚本
        # 或者 environment.yml 与脚本在同一目录
        if not os.path.exists(ENV_FILE_NAME):
             print(f"错误: {ENV_FILE_NAME} 在当前目录 {original_cwd} 和期望目录 {EXPECTED_PROJECT_ROOT} 中均未找到。")
             print(f"请确保 {ENV_FILE_NAME} 存在于你的项目根目录，并且脚本从此根目录运行，或者已正确设置 EXPECTED_PROJECT_ROOT。")
             return

    current_project_dir = os.getcwd() # 这是我们实际进行操作的目录
    print(f"将在此目录进行Conda环境操作: {current_project_dir}")

    # 2. 如果 'conda' 命令不可用，则安装 Miniconda
    conda_executable_path = "/usr/local/bin/conda"
    try:
        subprocess.run([conda_executable_path, "--version"], capture_output=True, check=True)
        print("Miniconda 已安装。")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Miniconda 未找到。正在安装 Miniconda...")
        try:
            subprocess.run(['wget', 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh', '-O', 'Miniconda3.sh'], check=True, timeout=60)
            subprocess.run(['chmod', '+x', 'Miniconda3.sh'], check=True)
            subprocess.run(['bash', './Miniconda3.sh', '-b', '-f', '-p', '/usr/local'], check=True, timeout=180)
            subprocess.run(['rm', 'Miniconda3.sh'], check=True)
            print("Miniconda 安装成功。")
            os.environ["PATH"] = f"/usr/local/bin:{os.environ['PATH']}"
        except subprocess.CalledProcessError as e:
            print(f"Miniconda 安装失败: {e}")
            if hasattr(e, 'stderr') and e.stderr: print(f"错误详情: {e.stderr.decode()}")
            return
        except subprocess.TimeoutExpired:
            print("Miniconda 安装超时。请检查网络连接或手动尝试安装。")
            return

    # 3. 检查 environment.yml 是否存在 (现在使用 ENV_FILE_NAME)
    env_file_path = os.path.join(current_project_dir, ENV_FILE_NAME) # 使用绝对路径
    if not os.path.exists(env_file_path):
        print(f"错误: 环境文件 {env_file_path} 未找到。")
        print(f"请确保 {ENV_FILE_NAME} 存在于目录 {current_project_dir} 中。")
        return
    print(f"找到环境配置文件: {env_file_path}")

    # 4. 创建或更新 Conda 环境
    env_name = "tsb_env"
    print(f"正在处理 Conda 环境 '{env_name}' (基于 {env_file_path})...")
    try:
        env_list_cmd = [conda_executable_path, 'env', 'list', '--json']
        result = subprocess.run(env_list_cmd, capture_output=True, text=True, check=True, timeout=30)
        env_data = json.loads(result.stdout)
        
        tsb_env_found = any(env_name in os.path.basename(env_path.rstrip('/')) for env_path in env_data['envs'])
        
        if tsb_env_found:
            print(f"环境 '{env_name}' 已存在。正在尝试更新 (这可能需要几分钟)...")
            update_cmd = [conda_executable_path, 'env', 'update', '--name', env_name, '--file', env_file_path, '--prune']
            subprocess.run(update_cmd, check=True, timeout=600)
            print(f"环境 '{env_name}' 更新完成。")
        else:
            print(f"环境 '{env_name}' 不存在。正在创建 (这可能需要几分钟)...")
            create_cmd = [conda_executable_path, 'env', 'create', '--name', env_name, '--file', env_file_path]
            subprocess.run(create_cmd, check=True, timeout=900)
            print(f"环境 '{env_name}' 创建成功。")

    except subprocess.CalledProcessError as e:
        print(f"Conda 环境设置过程中发生错误: {e}")
        print(f"执行的命令: {' '.join(e.cmd)}")
        # Safely access and decode stdout/stderr
        stdout_msg = e.stdout.decode() if hasattr(e.stdout, 'decode') and e.stdout is not None else str(e.stdout)
        stderr_msg = e.stderr.decode() if hasattr(e.stderr, 'decode') and e.stderr is not None else str(e.stderr)
        if stdout_msg: print(f"输出: {stdout_msg}")
        if stderr_msg: print(f"错误信息: {stderr_msg}")
        print(f"请检查您的 {env_file_path} 文件中的包名、版本号和channels是否正确。")
        print("特别是，确认所有列出的包（尤其是PyTorch的特定版本）在指定的channels中可用。")
        return
    except subprocess.TimeoutExpired:
        print("Conda 环境设置超时。这通常发生在下载大量包或网络连接慢的情况下。")
        print("你可以尝试增加脚本中的超时设置，或检查网络。")
        return
    except FileNotFoundError:
        print(f"错误: Conda 可执行文件 '{conda_executable_path}' 未找到。这不应该在Miniconda安装后发生。")
        return
    except json.JSONDecodeError:
        print(f"错误: 解析 'conda env list' 的JSON输出失败。conda的输出: {result.stdout if 'result' in locals() else 'N/A'}")
        return

    print("\nColab 环境初始化脚本执行完毕。")
    print("要在此环境中执行命令 (例如运行Python脚本)，请在Colab单元格中使用:")
    print(f"!source /usr/local/etc/profile.d/conda.sh && conda activate {env_name} && python your_script.py")
    print("或者，更直接地运行Python脚本:")
    print(f"!{conda_executable_path} run -n {env_name} python your_script.py")

if __name__ == '__main__':
    if 'google.colab' in sys.modules or os.getenv('COLAB_GPU') or os.getenv('TRAMPOLINE_VERSION'):
        initialize_colab_environment()
    else:
        print("此脚本设计用于Google Colab环境。") 