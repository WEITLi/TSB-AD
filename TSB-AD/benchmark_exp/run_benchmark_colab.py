import pandas as pd
import numpy as np
import torch
import random
import time
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
import networkx
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict, Optimal_Uni_algo_HP_dict

# 定义要使用的模型
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

def set_seed(seed=2024):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_model_size(model):
    """计算模型大小（MB）和参数数量"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb, sum(p.numel() for p in model.parameters())

def run_experiment(AD_Name, data_train, data, is_multivariate=True, n_runs=2):
    """运行多次实验并返回结果"""
    results = []
    inference_times = []
    model_sizes = []
    
    # 获取超参数
    Optimal_Det_HP = Optimal_Multi_algo_HP_dict[AD_Name] if is_multivariate else Optimal_Uni_algo_HP_dict[AD_Name]
    
    for run in range(n_runs):
        print(f"运行 {run + 1}/{n_runs}")
        
        # 记录推理时间
        start_time = time.time()
        
        # 运行模型
        if AD_Name in Semisupervise_AD_Pool:
            output = run_Semisupervise_AD(AD_Name, data_train, data, **Optimal_Det_HP)
        elif AD_Name in Unsupervise_AD_Pool:
            output = run_Unsupervise_AD(AD_Name, data, **Optimal_Det_HP)
        else:
            raise ValueError(f"{AD_Name} 未定义")
            
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # 获取模型大小（如果可用）
        try:
            if hasattr(output, 'model'):
                model_size, num_params = get_model_size(output.model)
                model_sizes.append((model_size, num_params))
        except:
            model_sizes.append((None, None))
            
        results.append(output)
    
    return results, inference_times, model_sizes

def evaluate_results(results, label, slidingWindow):
    """评估结果并计算所有指标"""
    metrics = []
    for output in results:
        if isinstance(output, np.ndarray):
            evaluation_result = get_metrics(output, label, metric='all', slidingWindow=slidingWindow)
            metrics.append(evaluation_result)
    
    # 计算平均值和标准差
    if metrics:
        mean_metrics = pd.DataFrame(metrics).mean()
        std_metrics = pd.DataFrame(metrics).std()
        return mean_metrics, std_metrics
    return None, None

def plot_results(df_results, metric='VUS-PR', title=None):
    """绘制结果箱线图"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_results, showfliers=False)
    plt.xticks(rotation=45)
    plt.title(title or f'{metric} 分布')
    plt.tight_layout()
    plt.show()

def friedman_nemenyi_test(df_perf, alpha=0.05):
    """执行Friedman和Nemenyi检验"""
    classifiers = df_perf['classifier_name'].unique()
    
    # Friedman检验
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1]
    
    if friedman_p_value >= alpha:
        print('无统计显著差异')
        return None, None, None
        
    # Nemenyi检验
    data = [df_perf.loc[df_perf['classifier_name'] == c]['accuracy'].values for c in classifiers]
    nemenyi = posthoc_nemenyi_friedman(np.array(data).T)
    
    # 计算平均排名
    df_ranks = df_perf.pivot(index='dataset_name', columns='classifier_name', values='accuracy')
    average_ranks = df_ranks.rank(ascending=False).mean(axis=0)
    
    return nemenyi, average_ranks, len(classifiers)

def main():
    # 设置随机种子
    set_seed()
    
    # 实验配置
    n_runs = 2  # 每个实验重复次数
    metrics_to_track = ['AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC', 
                       'Standard-F1', 'PA-F1', 'Event-based-F1', 
                       'R-based-F1', 'Affiliation-F']
    
    # 准备结果存储
    results = {
        'univariate': {},
        'multivariate': {}
    }
    
    # 处理单变量数据集
    print("处理单变量数据集...")
    uni_filename = '451_UCR_id_149_Medical_tr_3000_1st_7175.csv'
    print(f"处理文件: {uni_filename}")
    # 加载数据
    df = pd.read_csv(f'../Datasets/TSB-AD-U/{uni_filename}').dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    
    # 获取训练集
    train_index = int(uni_filename.split('.')[0].split('_')[-3])
    data_train = data[:train_index, :]
    
    # 对每个模型进行实验
    for AD_Name in uni_algorithms:
        print(f"运行模型: {AD_Name}")
        model_results, inference_times, model_sizes = run_experiment(
            AD_Name, data_train, data, is_multivariate=False, n_runs=n_runs
        )
        
        # 评估结果
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        mean_metrics, std_metrics = evaluate_results(model_results, label, slidingWindow)
        
        if mean_metrics is not None:
            results['univariate'][f"{AD_Name}_{uni_filename}"] = {
                'metrics': mean_metrics,
                'std': std_metrics,
                'inference_time': np.mean(inference_times),
                'model_size': model_sizes[0] if model_sizes else (None, None)
            }
    
    # 处理多变量数据集
    print("\n处理多变量数据集...")
    multi_filename = '178_Exathlon_id_5_Facility_tr_12538_1st_12638.csv'
    print(f"处理文件: {multi_filename}")
    # 加载数据
    df = pd.read_csv(f'../Datasets/TSB-AD-M/{multi_filename}').dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    
    # 获取训练集
    train_index = int(multi_filename.split('.')[0].split('_')[-3])
    data_train = data[:train_index, :]
    
    # 对每个模型进行实验
    for AD_Name in multi_algorithms:
        print(f"运行模型: {AD_Name}")
        model_results, inference_times, model_sizes = run_experiment(
            AD_Name, data_train, data, is_multivariate=True, n_runs=n_runs
        )
        
        # 评估结果
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        mean_metrics, std_metrics = evaluate_results(model_results, label, slidingWindow)
        
        if mean_metrics is not None:
            results['multivariate'][f"{AD_Name}_{multi_filename}"] = {
                'metrics': mean_metrics,
                'std': std_metrics,
                'inference_time': np.mean(inference_times),
                'model_size': model_sizes[0] if model_sizes else (None, None)
            }
    
    # 保存结果
    print("\n保存结果...")
    for dataset_type in ['univariate', 'multivariate']:
        # 保存详细结果
        detailed_results = pd.DataFrame()
        for key, value in results[dataset_type].items():
            model_name, filename = key.split('_', 1)
            row = {
                'model': model_name,
                'dataset': filename,
                **value['metrics'],
                **{f'{k}_std': v for k, v in value['std'].items()},
                'inference_time': value['inference_time'],
                'model_size_mb': value['model_size'][0],
                'num_parameters': value['model_size'][1]
            }
            detailed_results = detailed_results.append(row, ignore_index=True)
        
        detailed_results.to_csv(f'results_{dataset_type}_detailed.csv', index=False)
        
        # 保存汇总结果
        summary = detailed_results.groupby('model').agg({
            **{metric: ['mean', 'std'] for metric in metrics_to_track},
            'inference_time': ['mean', 'std'],
            'model_size_mb': 'first',
            'num_parameters': 'first'
        })
        summary.to_csv(f'results_{dataset_type}_summary.csv')
        
        # 绘制结果
        for metric in metrics_to_track:
            plot_results(
                detailed_results.pivot(index='dataset', columns='model', values=metric),
                metric=metric,
                title=f'{dataset_type.capitalize()} - {metric}'
            )

if __name__ == '__main__':
    main() 