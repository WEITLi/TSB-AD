# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

# --- 1. 导入所需库 ---
import pandas as pd  # 用于数据处理，特别是读取CSV文件
import numpy as np   # 用于数值计算
import torch         # PyTorch库，主要用于设置随机种子和检查CUDA
import random, argparse, time, os, logging # 标准库：随机数、命令行参数解析、时间、操作系统交互、日志记录
from TSB_AD.evaluation.metrics import get_metrics # TSB-AD框架：用于评估模型性能
from TSB_AD.utils.slidingWindows import find_length_rank # TSB-AD框架：可能用于确定滑动窗口大小
from TSB_AD.model_wrapper import * # TSB-AD框架：导入模型调用接口、模型池列表
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict # TSB-AD框架：导入预定义的多变量最优超参数

# --- 2. 设置随机种子以保证可复现性 ---
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False # 禁用cudnn的自动优化，确保确定性
torch.backends.cudnn.deterministic = True # 确保cudnn的操作是确定性的

# --- 3. 检查并打印CUDA环境信息 ---
print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

# --- 4. 主程序入口 ---
if __name__ == '__main__':

    Start_T = time.time() # 记录脚本开始时间

    # --- 4.1 解析命令行参数 ---
    parser = argparse.ArgumentParser(description='Generating Anomaly Score for Multivariate Datasets')
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/TSB-AD-M/', 
                        help='Directory containing the multivariate dataset files.') # 数据集目录
    parser.add_argument('--file_list', type=str, default='../Datasets/File_List/TSB-AD-M-Eva.csv', 
                        help='CSV file containing the list of filenames to process.') # 待处理文件列表
    parser.add_argument('--score_dir', type=str, default='eval/score/multi/', 
                        help='Directory to save the raw anomaly scores (.npy files).') # 异常得分保存目录
    parser.add_argument('--save_dir', type=str, default='eval/metrics/multi/', 
                        help='Directory to save the evaluation metrics (CSV file).') # 评估结果保存目录
    parser.add_argument('--save', type=bool, default=False, 
                        help='Whether to calculate and save evaluation metrics.') # 是否保存评估结果
    parser.add_argument('--AD_Name', type=str, default='IForest', 
                        help='Name of the anomaly detection algorithm to use.') # 指定要运行的算法名称
    args = parser.parse_args() # 解析参数

    # --- 4.2 设置输出目录和日志记录 ---
    # 构建特定算法的得分保存路径
    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok = True) # 创建目录（如果不存在）
    # 配置日志记录器，将日志写入特定算法的日志文件
    logging.basicConfig(filename=f'{target_dir}/000_run_{args.AD_Name}.log', 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # --- 4.3 加载文件列表和算法超参数 ---
    # 从CSV读取需要处理的文件名
    file_list = pd.read_csv(args.file_list)['file_name'].values 
    # 根据指定的算法名称，从预定义的字典中获取最优超参数
    Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name] 
    print('Optimal Hyperparameters: ', Optimal_Det_HP)
    logging.info(f'Using Optimal Hyperparameters: {Optimal_Det_HP}')

    # --- 4.4 循环处理每个数据集文件 ---
    write_csv = [] # 用于存储待写入CSV的评估结果
    for filename in file_list:
        # 检查得分文件是否已存在，若存在则跳过，避免重复计算
        score_file_path = os.path.join(target_dir, filename.split('.')[0] + '.npy')
        if os.path.exists(score_file_path):
            logging.info(f'Score file already exists for {filename}. Skipping.')
            continue
        
        print('Processing: {} using algorithm: {}'.format(filename, args.AD_Name))
        logging.info(f'Processing: {filename} using algorithm: {args.AD_Name}')

        # --- 4.4.1 加载和预处理数据 ---
        file_path = os.path.join(args.dataset_dir, filename) # 构建完整文件路径
        try:
            df = pd.read_csv(file_path).dropna() # 读取CSV并移除缺失值
            data = df.iloc[:, 0:-1].values.astype(float) # 提取特征数据
            label = df['Label'].astype(int).to_numpy() # 提取标签
        except FileNotFoundError:
            logging.error(f'Data file not found: {file_path}. Skipping.')
            continue
        except Exception as e:
            logging.error(f'Error loading or processing file {file_path}: {e}. Skipping.')
            continue
        
        # --- 4.4.2 计算辅助参数 (滑动窗口和训练集索引) ---
        # feats = data.shape[1] # 获取特征数量 (此脚本中未使用)
        # 使用第一个特征计算滑动窗口大小 (可能基于周期性)
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1) 
        # 从文件名中解析训练集大小 (依赖特定命名格式，如 ..._tr_INDEX_... 或 ..._INDEX_...)
        try:
            # 改进训练索引提取逻辑
            base_name = filename.split('.')[0]
            parts = base_name.split('_')
            train_index_str = None
            if 'tr' in parts:
                try:
                    tr_idx = parts.index('tr')
                    if tr_idx + 1 < len(parts) and parts[tr_idx + 1].isdigit():
                         train_index_str = parts[tr_idx + 1]
                except ValueError:
                     pass # 'tr' not in parts
            if train_index_str is None: # Fallback to last digit group if 'tr' pattern fails
                 train_index_str = next((p for p in reversed(parts) if p.isdigit()), None)
            
            if train_index_str is None:
                raise ValueError("Cannot extract training index from filename")
            train_index = int(train_index_str)
            if not (0 < train_index < len(data)):
                 raise ValueError(f"Extracted training index {train_index} is out of bounds for data length {len(data)}")
            data_train = data[:train_index, :] # 创建训练集
        except Exception as e:
            logging.error(f'Error extracting train index from {filename}: {e}. Skipping file.')
            continue

        # --- 4.4.3 运行指定的异常检测算法 ---
        start_time = time.time() # 记录模型运行开始时间
        output = None # 初始化output变量
        try:
            # 根据算法类型（半监督/无监督）调用对应的model_wrapper函数
            if args.AD_Name in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)
            elif args.AD_Name in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
            else:
                # 如果算法未在任何已知池中定义，记录错误
                logging.error(f"Algorithm '{args.AD_Name}' is not defined in Semisupervise_AD_Pool or Unsupervise_AD_Pool.")
                continue # 跳过此文件
        except Exception as e:
             logging.error(f"Error running algorithm {args.AD_Name} on {filename}: {e}")
             continue # 运行出错，跳过此文件

        end_time = time.time() # 记录模型运行结束时间
        run_time = end_time - start_time # 计算运行时间

        # --- 4.4.4 处理和保存模型输出 (异常得分) ---
        if isinstance(output, np.ndarray):
            # 如果成功返回得分数组
            logging.info(f'Success: {filename} using {args.AD_Name} | Time: {run_time:.3f}s | Length: {len(label)}')
            np.save(score_file_path, output) # 保存得分到 .npy 文件
        else:
            # 如果运行失败或返回错误信息
            logging.error(f'Failure or invalid output for {filename} using {args.AD_Name}: {output}')
            continue # 跳过评估和保存

        # --- 4.4.5 (可选) 计算并保存评估指标 ---
        if args.save:
            try:
                # 调用 TSB_AD 的评估函数计算指标
                evaluation_result = get_metrics(output, label, metric='all', slidingWindow=slidingWindow)
                print('Evaluation Result: ', evaluation_result)
                list_w = list(evaluation_result.values()) # 获取指标值
            except Exception as e:
                # 如果评估出错，记录错误并使用默认值填充
                logging.warning(f'Could not evaluate metrics for {filename}: {e}')
                list_w = [0]*len(get_metrics.__defaults__[0]) # 使用get_metrics中定义的默认指标数量
            
            # 将运行时间和文件名添加到结果列表
            list_w.insert(0, run_time)
            list_w.insert(0, filename)
            write_csv.append(list_w) # 添加到待写入CSV的总列表

            # --- 临时保存评估结果到CSV (每次处理完一个文件后) ---
            # 防止脚本中途失败丢失已处理结果
            temp_save_path = os.path.join(args.save_dir, f'{args.AD_Name}.csv')
            os.makedirs(args.save_dir, exist_ok=True) # 确保保存目录存在
            try:
                col_w = list(evaluation_result.keys()) # 获取指标名称作为列名
            except NameError: # 如果 evaluation_result 未成功计算
                 # 需要定义默认的列名，或者从get_metrics获取
                 # 假设 get_metrics 总是返回一个字典，即使在错误时也是如此，但这取决于其实现
                 # 或者定义一个标准的列名列表
                 logging.warning("Cannot determine metric names for CSV header due to evaluation error.")
                 # 使用一个预定义的或基于默认数量的列名列表
                 default_metric_names = [f'metric_{i+1}' for i in range(len(list_w) - 2)] # 减去文件和时间列
                 col_w = default_metric_names

            col_w.insert(0, 'Time') # 添加 'Time' 列
            col_w.insert(0, 'file') # 添加 'file' 列
            w_csv = pd.DataFrame(write_csv, columns=col_w) # 创建DataFrame
            w_csv.to_csv(temp_save_path, index=False) # 写入CSV
            logging.info(f'Temporarily saved evaluation results to {temp_save_path}')

    logging.info(f"Finished processing all files for algorithm {args.AD_Name}. Total time: {time.time() - Start_T:.3f}s")
    print(f"Finished processing all files for algorithm {args.AD_Name}. Total time: {time.time() - Start_T:.3f}s")