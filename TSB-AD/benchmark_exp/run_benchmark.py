# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, sys, logging
from datetime import datetime
import json

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
def set_seed(seed_value):
    """设置随机种子函数，用于多次实验"""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed_value

# 初始种子
seed = 2024
set_seed(seed)

def run_single_benchmark(dataset_type, dataset_filename, dataset_dir, algorithms, optimal_hp_dict, save_dir, run_id=1):
    """运行指定类型数据集上的基准测试"""
    print(f"\n--- Running Benchmark for {dataset_type} dataset: {dataset_filename} (Run {run_id}) ---")
    logging.info(f"--- Running Benchmark for {dataset_type} dataset: {dataset_filename} (Run {run_id}) ---")
    
    results_list = []
    model_details_dict = {}  # 用于存储模型参数和大小信息
    file_path = os.path.abspath(os.path.join(dataset_dir, dataset_filename))
    
    try:
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}. Skipping benchmark for {dataset_type}.")
        print(f"错误: 数据文件未找到 {file_path}. 跳过 {dataset_type} 基准测试。")
        return None, None
    except Exception as e:
        logging.error(f"Error loading data for {dataset_filename}: {e}")
        print(f"错误: 加载数据 {dataset_filename} 时出错: {e}")
        return None, None
        
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
        model_details = None
        run_status = "Success"
        error_msg = ""
        
        try:
            if algo_name in Semisupervise_AD_Pool:
                if data_train is None:
                     raise ValueError("Training data is required for semi-supervised models but could not be extracted.")
                if 'return_model_details' in optimal_hp:
                    optimal_hp_copy = optimal_hp.copy()  # 创建参数副本避免修改原始字典
                    optimal_hp_copy['return_model_details'] = True  # 设置为返回模型详细信息
                    output, model_details = run_Semisupervise_AD(algo_name, data_train, data, return_model_details=True, **optimal_hp_copy)
                else:
                    output, model_details = run_Semisupervise_AD(algo_name, data_train, data, return_model_details=True, **optimal_hp)
            elif algo_name in Unsupervise_AD_Pool:
                if 'return_model_details' in optimal_hp:
                    optimal_hp_copy = optimal_hp.copy()  # 创建参数副本避免修改原始字典
                    optimal_hp_copy['return_model_details'] = True  # 设置为返回模型详细信息
                    output, model_details = run_Unsupervise_AD(algo_name, data, return_model_details=True, **optimal_hp_copy)
                else:
                    output, model_details = run_Unsupervise_AD(algo_name, data, return_model_details=True, **optimal_hp)
            else:
                 raise ValueError(f"Algorithm '{algo_name}' not found in known pools.")
        except Exception as e:
            run_status = "Failure"
            error_msg = str(e)
            logging.error(f"Error running {algo_name} on {dataset_filename}: {e}")
            print(f"错误: 运行 {algo_name} 在 {dataset_filename} 上失败: {e}")
        
        run_time = time.time() - start_time
        
        # 保存模型详细信息
        if model_details is not None:
            model_details_dict[algo_name] = model_details
            print(f"  Collected model details for {algo_name}: {len(model_details)} properties")
        
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
            'Time': run_time,
            'Run': run_id,
        }
        
        # 添加模型参数和大小信息，如果有的话
        if model_details is not None:
            result_row['TotalParams'] = model_details.get('total_params', 'NA')
            result_row['ModelSizeMB'] = model_details.get('model_size_MB', 'NA')
            result_row['TrainableParams'] = model_details.get('trainable_params', 'NA')
        
        result_row.update(evaluation_result)
        results_list.append(result_row)

    # 保存结果
    if results_list:
        df_results = pd.DataFrame(results_list)
        # 确保列顺序一致
        # 使用默认指标名称列表确保所有列都存在
        cols = ['Dataset', 'Type', 'Algorithm', 'Status', 'Run', 'Time', 'TotalParams', 
                'ModelSizeMB', 'TrainableParams', 'ErrorMsg'] + DEFAULT_METRIC_NAMES
        # 过滤掉可能不存在的列 (以防万一，虽然现在应该不会发生)
        cols = [c for c in cols if c in df_results.columns]
        df_results = df_results[cols]
        
        save_path = os.path.abspath(os.path.join(save_dir, f"benchmark_results_{dataset_type}_run{run_id}.csv"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_results.to_csv(save_path, index=False)
        print(f"Benchmark results for {dataset_type} (Run {run_id}) saved to {save_path}")
        logging.info(f"Benchmark results for {dataset_type} (Run {run_id}) saved to {save_path}")
        
        # 保存模型详细信息
        if model_details_dict:
            model_details_save_path = os.path.abspath(os.path.join(save_dir, f"model_details_{dataset_type}_run{run_id}.json"))
            try:
                # 转换不可序列化的值
                serializable_details = {}
                for model_name, details in model_details_dict.items():
                    serializable_model = {}
                    for k, v in details.items():
                        if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                            serializable_model[k] = v
                        else:
                            serializable_model[k] = str(v)
                    serializable_details[model_name] = serializable_model
                
                with open(model_details_save_path, 'w') as f:
                    json.dump(serializable_details, f, indent=4)
                print(f"Model details saved to {model_details_save_path}")
                logging.info(f"Model details saved to {model_details_save_path}")
            except Exception as e:
                print(f"Error saving model details: {e}")
                logging.error(f"Error saving model details: {e}")
                
        return df_results, model_details_dict
    else:
        print(f"No results generated for {dataset_type}.")
        logging.warning(f"No results generated for {dataset_type}.")
        return None, None

def combine_run_results(save_dir, dataset_type, num_runs):
    """合并多次运行的结果"""
    all_runs = []
    
    for run_id in range(1, num_runs + 1):
        file_path = os.path.join(save_dir, f"benchmark_results_{dataset_type}_run{run_id}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_runs.append(df)
            print(f"Loaded results from run {run_id} for {dataset_type}")
        else:
            print(f"Warning: Results file for run {run_id} not found at {file_path}")
    
    if not all_runs:
        print(f"No run results found for {dataset_type}")
        return None
        
    combined_df = pd.concat(all_runs, ignore_index=True)
    combined_path = os.path.join(save_dir, f"benchmark_results_{dataset_type}_combined.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined {len(all_runs)} runs for {dataset_type}, saved to {combined_path}")
    
    return combined_df

def calculate_statistics(combined_df, save_dir, dataset_type):
    """计算统计信息（平均值、标准差等）"""
    if combined_df is None or len(combined_df) == 0:
        print(f"No data available for {dataset_type} to calculate statistics")
        return None
        
    # 按 Dataset 和 Algorithm 分组
    grouped = combined_df.groupby(['Dataset', 'Algorithm'])
    
    # 准备保存统计结果
    stats_rows = []
    
    # 遍历每个组，计算统计量
    for (dataset, algo), group in grouped:
        # 基本信息
        row = {
            'Dataset': dataset,
            'Algorithm': algo,
            'Type': dataset_type,
            'NumRuns': len(group)
        }
        
        # 计算执行时间统计
        row['Time_Mean'] = group['Time'].mean()
        row['Time_Std'] = group['Time'].std()
        
        # 计算模型参数和大小统计（如果存在）
        if 'TotalParams' in group.columns and 'NA' not in group['TotalParams'].values:
            try:
                row['TotalParams'] = group['TotalParams'].mean()
            except:
                row['TotalParams'] = 'NA'
                
        if 'ModelSizeMB' in group.columns and 'NA' not in group['ModelSizeMB'].values:
            try:
                row['ModelSizeMB'] = group['ModelSizeMB'].mean()
            except:
                row['ModelSizeMB'] = 'NA'
        
        # 计算所有指标的统计量
        for metric in DEFAULT_METRIC_NAMES:
            if metric in group.columns:
                # 过滤掉非数值值
                numeric_values = pd.to_numeric(group[metric], errors='coerce').dropna()
                if len(numeric_values) > 0:
                    row[f'{metric}_Mean'] = numeric_values.mean()
                    row[f'{metric}_Std'] = numeric_values.std()
                else:
                    row[f'{metric}_Mean'] = 'NA'
                    row[f'{metric}_Std'] = 'NA'
        
        stats_rows.append(row)
    
    # 创建统计数据框
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_path = os.path.join(save_dir, f"benchmark_stats_{dataset_type}.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Statistics calculated for {dataset_type}, saved to {stats_path}")
        return stats_df
    else:
        print(f"No statistics calculated for {dataset_type}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Benchmark Evaluation')
    parser.add_argument('--uni_dataset', type=str, default='451_UCR_id_149_Medical_tr_3000_1st_7175.csv', help='Univariate dataset filename.')
    parser.add_argument('--multi_dataset', type=str, default='178_Exathlon_id_5_Facility_tr_12538_1st_12638.csv', help='Multivariate dataset filename.')
    parser.add_argument('--uni_dataset_dir', type=str, default='../Datasets/TSB-AD-U/', help='Directory for univariate datasets.')
    parser.add_argument('--multi_dataset_dir', type=str, default='../Datasets/TSB-AD-M/', help='Directory for multivariate datasets.')
    parser.add_argument('--save_dir', type=str, default='eval/benchmark/', help='Directory to save benchmark results.')
    parser.add_argument('--log_file', type=str, default='eval/benchmark/benchmark_run.log', help='Path to the log file.')
    parser.add_argument('--num_runs', type=int, default=2, help='Number of runs for each experiment (for stability analysis).')
    parser.add_argument('--dataset_type', type=str, choices=['Univariate', 'Multivariate', 'Both'], default='Both', help='Type of dataset to benchmark.')

    args = parser.parse_args()

    # --- 配置日志 --- 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_save_path = os.path.abspath(os.path.join(script_dir, args.log_file.replace('.log', f'_{timestamp}.log')))
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

    # --- 定义数据集路径 --- 
    uni_dir = os.path.abspath(os.path.join(script_dir, args.uni_dataset_dir))
    multi_dir = os.path.abspath(os.path.join(script_dir, args.multi_dataset_dir))
    save_dir = os.path.abspath(os.path.join(script_dir, args.save_dir))
    
    # --- 运行基准测试 --- 
    print(f"--- Running Benchmark for specified algorithms ({args.num_runs} runs each) ---")
    logging.info(f"--- Running Benchmark for specified algorithms ({args.num_runs} runs each) ---")

    # 根据选择的数据集类型运行实验
    uni_results_all_runs = []
    multi_results_all_runs = []
    
    if args.dataset_type in ['Univariate', 'Both']:
        print("\n=== Running Univariate Benchmarks ===")
        logging.info("=== Running Univariate Benchmarks ===")
        for run_id in range(1, args.num_runs + 1):
            # 为每次运行设置不同的种子
            current_seed = seed + run_id
            set_seed(current_seed)
            print(f"\n--- Run {run_id}/{args.num_runs} with seed {current_seed} ---")
            logging.info(f"--- Run {run_id}/{args.num_runs} with seed {current_seed} ---")
            
            uni_results, uni_model_details = run_single_benchmark(
                'Univariate', args.uni_dataset, uni_dir, 
                uni_algorithms, Optimal_Uni_algo_HP_dict, save_dir, run_id
            )
            uni_results_all_runs.append(uni_results)
        
        # 合并并计算统计信息
        uni_combined = combine_run_results(save_dir, 'Univariate', args.num_runs)
        uni_stats = calculate_statistics(uni_combined, save_dir, 'Univariate')

    if args.dataset_type in ['Multivariate', 'Both']:
        print("\n=== Running Multivariate Benchmarks ===")
        logging.info("=== Running Multivariate Benchmarks ===")
        for run_id in range(1, args.num_runs + 1):
            # 为每次运行设置不同的种子
            current_seed = seed + 100 + run_id  # 与单变量实验使用不同的种子范围
            set_seed(current_seed)
            print(f"\n--- Run {run_id}/{args.num_runs} with seed {current_seed} ---")
            logging.info(f"--- Run {run_id}/{args.num_runs} with seed {current_seed} ---")
            
            multi_results, multi_model_details = run_single_benchmark(
                'Multivariate', args.multi_dataset, multi_dir, 
                multi_algorithms, Optimal_Multi_algo_HP_dict, save_dir, run_id
            )
            multi_results_all_runs.append(multi_results)
            
        # 合并并计算统计信息
        multi_combined = combine_run_results(save_dir, 'Multivariate', args.num_runs)
        multi_stats = calculate_statistics(multi_combined, save_dir, 'Multivariate')

    print("\nAll benchmark runs completed.")
    logging.info("All benchmark runs completed.")
    
    # 输出总结信息
    if args.dataset_type in ['Univariate', 'Both'] and uni_results_all_runs:
        successful_uni_runs = sum(1 for r in uni_results_all_runs if r is not None)
        print(f"Univariate: {successful_uni_runs}/{args.num_runs} successful runs")
        
    if args.dataset_type in ['Multivariate', 'Both'] and multi_results_all_runs:
        successful_multi_runs = sum(1 for r in multi_results_all_runs if r is not None)
        print(f"Multivariate: {successful_multi_runs}/{args.num_runs} successful runs")
        
    print(f"\nAll results saved to {save_dir}")
    print("Benchmark finished.")
    logging.info("Benchmark finished.") 