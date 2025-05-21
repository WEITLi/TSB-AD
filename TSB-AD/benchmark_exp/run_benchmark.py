# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, sys, logging

# --- 手动添加项目根目录到 sys.path (确保导入正常) ---
script_dir = os.path.dirname(os.path.abspath(__file__)) # benchmark_exp 目录
project_root = os.path.abspath(os.path.join(script_dir, '..')) # TSB-AD/TSB-AD 目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"--- Debug: Added to sys.path: {project_root}")
else:
    print(f"--- Debug: {project_root} already in sys.path.")
# ------------------------------------

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD, Semisupervise_AD_Pool, Unsupervise_AD_Pool
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict

# --- 获取默认指标名称 (更健壮的方式) ---
def get_default_metric_names():
    # 尝试创建一个虚拟输入来获取列名，或者直接硬编码
    # 硬编码已知指标是一个简单且可靠的选择
    return ['AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC', 'Standard-F1', 'PA-F1', 'Event-based-F1', 'R-based-F1', 'Affiliation-F']
DEFAULT_METRIC_NAMES = get_default_metric_names()
# ------------------------------------

# seeding (保持与 TSB-AD 一致)
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def run_single_benchmark(dataset_type, dataset_filename, dataset_dir, algorithms, optimal_hp_dict, save_dir):
    """运行指定类型数据集上的基准测试"""
    print(f"\n--- Running Benchmark for {dataset_type} dataset: {dataset_filename} ---")
    logging.info(f"--- Running Benchmark for {dataset_type} dataset: {dataset_filename} ---")
    
    results_list = []
    file_path = os.path.abspath(os.path.join(dataset_dir, dataset_filename))
    
    try:
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}. Skipping benchmark for {dataset_type}.")
        print(f"错误: 数据文件未找到 {file_path}. 跳过 {dataset_type} 基准测试。")
        return None
    except Exception as e:
        logging.error(f"Error loading data for {dataset_filename}: {e}")
        print(f"错误: 加载数据 {dataset_filename} 时出错: {e}")
        return None
        
    # 提取训练索引 (如果需要)
    data_train = None
    try:
        base_name = dataset_filename.split('.')[0]
        parts = base_name.split('_')
        train_index_str = None
        if 'tr' in parts:
            try:
                tr_idx = parts.index('tr')
                if tr_idx + 1 < len(parts) and parts[tr_idx + 1].isdigit():
                    train_index_str = parts[tr_idx + 1]
            except ValueError:
                pass
        if train_index_str is None:
            train_index_str = next((p for p in reversed(parts) if p.isdigit()), None)
            
        if train_index_str is None:
            raise ValueError("Cannot extract training index from filename")
        train_index = int(train_index_str)
        if not (0 < train_index < len(data)):
            raise ValueError(f"Extracted training index {train_index} is out of bounds")
        data_train = data[:train_index, :]
    except Exception as e:
        logging.warning(f"Could not extract train index for {dataset_filename}: {e}. Semi-supervised models might fail.")
        print(f"警告: 无法从 {dataset_filename} 提取训练索引: {e}. 半监督模型可能失败。")
        
    # 计算 slidingWindow
    slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
    
    # 运行算法
    for algo_name in algorithms:
        print(f"Running {algo_name} on {dataset_filename}...")
        logging.info(f"Running {algo_name} on {dataset_filename}...")
        
        if algo_name not in optimal_hp_dict:
            logging.warning(f"Optimal hyperparameters not found for {algo_name}. Skipping.")
            print(f"警告: 未找到 {algo_name} 的最优超参数，跳过。")
            continue
            
        optimal_hp = optimal_hp_dict[algo_name]
        start_time = time.time()
        output = None
        run_status = "Success"
        error_msg = ""
        
        try:
            if algo_name in Semisupervise_AD_Pool:
                if data_train is None:
                     raise ValueError("Training data is required for semi-supervised models but could not be extracted.")
                output = run_Semisupervise_AD(algo_name, data_train, data, **optimal_hp)
            elif algo_name in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(algo_name, data, **optimal_hp)
            else:
                 raise ValueError(f"Algorithm '{algo_name}' not found in known pools.")
        except Exception as e:
            run_status = "Failure"
            error_msg = str(e)
            logging.error(f"Error running {algo_name} on {dataset_filename}: {e}")
            print(f"错误: 运行 {algo_name} 在 {dataset_filename} 上失败: {e}")
        
        run_time = time.time() - start_time
        
        # 评估 (即使失败也要记录)
        evaluation_result = {}
        if run_status == "Success" and isinstance(output, np.ndarray):
            try:
                # 移除 metric='all' 参数
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                print(f"  {algo_name} Evaluation: {evaluation_result}")
            except Exception as e:
                logging.warning(f"Could not evaluate metrics for {algo_name} on {dataset_filename}: {e}")
                print(f"  警告: 无法评估 {algo_name} 在 {dataset_filename} 上的指标: {e}")
                # 记录评估错误，但仍继续
                # 使用预定义的默认指标名称列表
                evaluation_result = {m: 'EvalError' for m in DEFAULT_METRIC_NAMES}
        elif run_status == "Failure":
             # 使用预定义的默认指标名称列表
             evaluation_result = {m: 'RunError' for m in DEFAULT_METRIC_NAMES}
        else: # output 不是 ndarray
             run_status = "InvalidOutput"
             error_msg = f"Invalid output type: {type(output)}"
             logging.error(f"Invalid output for {algo_name} on {dataset_filename}: {type(output)}")
             # 使用预定义的默认指标名称列表
             evaluation_result = {m: 'RunError' for m in DEFAULT_METRIC_NAMES}

        # 准备结果行
        result_row = {
            'Dataset': dataset_filename,
            'Type': dataset_type,
            'Algorithm': algo_name,
            'Status': run_status,
            'ErrorMsg': error_msg,
            'Time': run_time
        }
        result_row.update(evaluation_result)
        results_list.append(result_row)

    # 保存结果
    if results_list:
        df_results = pd.DataFrame(results_list)
        # 确保列顺序一致
        # 使用默认指标名称列表确保所有列都存在
        cols = ['Dataset', 'Type', 'Algorithm', 'Status', 'Time', 'ErrorMsg'] + DEFAULT_METRIC_NAMES
        # 过滤掉可能不存在的列 (以防万一，虽然现在应该不会发生)
        cols = [c for c in cols if c in df_results.columns]
        df_results = df_results[cols]
        
        save_path = os.path.abspath(os.path.join(save_dir, f"benchmark_results_{dataset_type}.csv"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_results.to_csv(save_path, index=False)
        print(f"Benchmark results for {dataset_type} saved to {save_path}")
        logging.info(f"Benchmark results for {dataset_type} saved to {save_path}")
        return df_results
    else:
        print(f"No results generated for {dataset_type}.")
        logging.warning(f"No results generated for {dataset_type}.")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Benchmark Evaluation')
    parser.add_argument('--uni_dataset', type=str, default='451_UCR_id_149_Medical_tr_3000_1st_7175.csv', help='Univariate dataset filename.')
    parser.add_argument('--multi_dataset', type=str, default='178_Exathlon_id_5_Facility_tr_12538_1st_12638.csv', help='Multivariate dataset filename.')
    parser.add_argument('--uni_dataset_dir', type=str, default='../Datasets/TSB-AD-U/', help='Directory for univariate datasets.')
    parser.add_argument('--multi_dataset_dir', type=str, default='../Datasets/TSB-AD-M/', help='Directory for multivariate datasets.')
    parser.add_argument('--save_dir', type=str, default='eval/benchmark/', help='Directory to save benchmark results.')
    parser.add_argument('--log_file', type=str, default='eval/benchmark/benchmark_run.log', help='Path to the log file.')

    args = parser.parse_args()

    # --- 配置日志 --- 
    log_save_path = os.path.abspath(os.path.join(script_dir, args.log_file))
    os.makedirs(os.path.dirname(log_save_path), exist_ok=True)
    logging.basicConfig(filename=log_save_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"Logging to {log_save_path}")

    # --- 定义算法列表 --- 
    uni_algorithms = [
        # 重构型
        'AutoEncoder', 'PCA', 'USAD', 'OmniAnomaly',
        # 预测型
        'LSTMAD', 'DLinear',
        # 插补
        'Donut',
        # 聚类
        'KMeansAD', 'LOF', 'OCSVM',
        # 混合
        'AnomalyTransformer'
        
    ]
    multi_algorithms = [
        # 重构型
        'AutoEncoder', 'PCA', 'USAD', 'OmniAnomaly',
        # 预测型
        'LSTMAD', 'DLinear',
        # 插补
        'Donut', 
        # 聚类
        'KMeansAD', 'LOF', 'OCSVM',
        # 混合
        'AnomalyTransformer'
        
    ]
    # uni_algorithms = ['DLinear']
    # multi_algorithms = ['DLinear']

    # --- 定义数据集路径 --- 
    uni_dir = os.path.abspath(os.path.join(script_dir, args.uni_dataset_dir))
    multi_dir = os.path.abspath(os.path.join(script_dir, args.multi_dataset_dir))
    save_dir = os.path.abspath(os.path.join(script_dir, args.save_dir))
    
    # --- 运行基准测试 --- 
    print("--- Running Benchmark for specified algorithms ---")
    logging.info("--- Running Benchmark for specified algorithms ---")
    uni_results = run_single_benchmark('Univariate', args.uni_dataset, uni_dir, uni_algorithms, Optimal_Uni_algo_HP_dict, save_dir)
    multi_results = run_single_benchmark('Multivariate', args.multi_dataset, multi_dir, multi_algorithms, Optimal_Multi_algo_HP_dict, save_dir)
    
    # --- 合并结果 (可选) ---
    if uni_results is not None and multi_results is not None:
        all_results = pd.concat([uni_results, multi_results], ignore_index=True)
        final_save_path = os.path.abspath(os.path.join(save_dir, 'benchmark_results_all.csv'))
        all_results.to_csv(final_save_path, index=False)
        print(f"Combined benchmark results saved to {final_save_path}")
        logging.info(f"Combined benchmark results saved to {final_save_path}")
    elif uni_results is not None:
         print("Only univariate results were generated.")
    elif multi_results is not None:
         print("Only multivariate results were generated.")
    else:
         print("No benchmark results were generated.")
         
    print("Benchmark finished.")
    logging.info("Benchmark finished.") 