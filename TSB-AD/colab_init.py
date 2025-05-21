import os
import subprocess
import sys
import json

def initialize_colab_environment():
    print("开始 Colab 环境初始化...")

    # 1. 确保在项目根目录 (Google Drive挂载和目录切换应在此脚本运行前完成)
    PROJECT_ROOT = "/content/drive/MyDrive/TSB-AD/TSB-AD" #你的项目路径
    if os.getcwd() != PROJECT_ROOT:
        if os.path.exists(PROJECT_ROOT):
            os.chdir(PROJECT_ROOT)
            print(f"已切换到项目目录: {PROJECT_ROOT}")
        else:
            print(f"错误: 项目根目录 {PROJECT_ROOT} 未找到。")
            print("请在运行此脚本前，确保 Google Drive 已挂载并且你已导航到正确的项目目录。")
            return

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
            # 使用 -u 禁止更新conda基础环境的包，加快安装速度
            subprocess.run(['bash', './Miniconda3.sh', '-b', '-f', '-p', '/usr/local'], check=True, timeout=180)
            subprocess.run(['rm', 'Miniconda3.sh'], check=True)
            print("Miniconda 安装成功。")
            # 如果刚安装，确保conda在PATH中对后续此脚本内的调用可见
            os.environ["PATH"] = f"/usr/local/bin:{os.environ['PATH']}"
        except subprocess.CalledProcessError as e:
            print(f"Miniconda 安装失败: {e}")
            if e.stderr: print(f"错误详情: {e.stderr.decode()}")
            return
        except subprocess.TimeoutExpired:
            print("Miniconda 安装超时。请检查网络连接或手动尝试安装。")
            return


    # 3. 检查 environment.yml 是否存在
    env_file = "environment.yml"
    if not os.path.exists(env_file):
        print(f"错误: {env_file} 在目录 {os.getcwd()} 中未找到。")
        print(f"请将 environment.yml 放置在项目根目录 {PROJECT_ROOT}。")
        # (你可以选择在此处自动创建默认的 environment.yml，如果这是期望的行为)
        return

    # 4. 创建或更新 Conda 环境
    env_name = "tsb_env"
    print(f"正在处理 Conda 环境 '{env_name}' (基于 {env_file})...")
    try:
        # 检查环境是否已存在
        env_list_cmd = [conda_executable_path, 'env', 'list', '--json']
        result = subprocess.run(env_list_cmd, capture_output=True, text=True, check=True, timeout=30)
        env_data = json.loads(result.stdout)
        
        # 注意：conda env list的路径可能因系统而异，检查名称部分
        tsb_env_found = False
        for env_path_full in env_data['envs']:
            if env_name in os.path.basename(env_path_full.rstrip('/')): # 更可靠的名称检查
                 tsb_env_found = True
                 break
        
        if tsb_env_found:
            print(f"环境 '{env_name}' 已存在。正在尝试更新 (这可能需要几分钟)...")
            update_cmd = [conda_executable_path, 'env', 'update', '--name', env_name, '--file', env_file, '--prune']
            subprocess.run(update_cmd, check=True, timeout=600) # 增加超时时间
            print(f"环境 '{env_name}' 更新完成。")
        else:
            print(f"环境 '{env_name}' 不存在。正在创建 (这可能需要几分钟)...")
            create_cmd = [conda_executable_path, 'env', 'create', '--name', env_name, '--file', env_file]
            subprocess.run(create_cmd, check=True, timeout=900) # 增加超时时间
            print(f"环境 '{env_name}' 创建成功。")

    except subprocess.CalledProcessError as e:
        print(f"Conda 环境设置过程中发生错误: {e}")
        print(f"执行的命令: {' '.join(e.cmd)}")
        if e.stdout: print(f"输出: {e.stdout}") # stdout可能是bytes
        if e.stderr: print(f"错误信息: {e.stderr}") # stderr可能是bytes
        print(f"请检查您的 {env_file} 文件中的包名、版本号和channels是否正确。")
        print("特别是，确认所有列出的包（尤其是PyTorch的特定版本）在指定的channels中可用。")
        return
    except subprocess.TimeoutExpired:
        print("Conda 环境设置超时。这通常发生在下载大量包或网络连接慢的情况下。")
        print("你可以尝试增加脚本中的超时设置，或检查网络。")
        return
    except FileNotFoundError: # conda_executable_path
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
    # 此脚本设计为在Colab中通过 `!python colab_init.py` 运行
    # `google.colab` 模块的检查主要用于避免在非Colab环境中意外执行
    if 'google.colab' in sys.modules or os.getenv('COLAB_GPU') or os.getenv('TRAMPOLINE_VERSION'):
        initialize_colab_environment()
    else:
        print("此脚本设计用于Google Colab环境。")
        # 你可以选择在这里添加本地执行的逻辑（如果需要）
        # initialize_colab_environment() 