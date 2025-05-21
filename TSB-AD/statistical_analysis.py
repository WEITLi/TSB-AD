import numpy as np
import pandas as pd
import math
from scipy.stats import friedmanchisquare
from scipy.stats import rankdata

def posthoc_nemenyi_friedman(data):
    """
    实现 Nemenyi 事后检验，替代 scikit_posthocs.posthoc_nemenyi_friedman
    
    参数:
    - data: numpy 数组，形状为 (k, n)，其中 k 是算法数量，n 是数据集数量
    
    返回:
    - p_values: DataFrame，包含成对 p 值
    """
    k, n = data.shape  # k 是算法数量，n 是数据集数量
    
    # 计算每个数据集的排名 (行内排名)
    ranks = np.zeros_like(data, dtype=float)
    for i in range(n):
        ranks[:, i] = rankdata(-data[:, i], method='average')  # 降序排名 (值越高排名越好)
    
    # 计算平均排名
    mean_ranks = ranks.mean(axis=1)
    
    # 创建空的 p 值矩阵 (算法 x 算法)
    p_values = np.ones((k, k))
    
    # 计算临界值 q_alpha
    # 这是 Nemenyi 检验的标准算法
    # 对于 alpha=0.05，使用近似值
    alpha = 0.05
    q_alpha = 1.960  # 0.05 显著性水平的近似 q 值
    
    # 计算标准差
    std_dev = np.sqrt(k * (k + 1) / (6 * n))
    
    # 计算所有算法对的 p 值
    for i in range(k):
        for j in range(i+1, k):
            # 计算排名差异
            diff = abs(mean_ranks[i] - mean_ranks[j])
            
            # 计算 z 分数
            z = diff / std_dev
            
            # 使用正态分布计算 p 值 (近似值)
            # 简化为双侧测试
            from scipy.stats import norm
            p_val = 2 * (1 - norm.cdf(z))
            
            # 调整 p 值 (Bonferroni-Dunn 调整)
            p_val_adj = min(p_val * k * (k - 1) / 2, 1.0)  # 不超过 1
            
            p_values[i, j] = p_val_adj
            p_values[j, i] = p_val_adj
    
    # 将结果转换为 DataFrame 格式，保持与 scikit_posthocs 相同的输出格式
    df_p_values = pd.DataFrame(p_values)
    
    return df_p_values

def Friedman_Nemenyi(alpha=0.05, df_perf=None):
    """执行 Friedman 检验和 Nemenyi 事后分析"""
    # 检查性能数据框是否存在
    if df_perf is None:
        return None, None, None
    
    # 按分类器名称分组，计算每个分类器的数据集数量
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    
    # 记录最大数据集数量
    max_nb_datasets = df_counts['count'].max()
    
    # 创建分类器列表（仅包含在所有数据集上均有结果的分类器）
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]['classifier_name'])
    
    # Friedman 检验需要每个分类器在每个数据集上的性能数据
    # 计算 Friedman 检验的 p 值
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1]
    
    # 如果 p 值大于 alpha，则无法拒绝零假设（无统计差异）
    if friedman_p_value >= alpha:
        print('无统计学差异...')
        return None, None, None
        
    # Friedman 检验通过，准备 Nemenyi 检验的输入数据
    data = []
    for c in classifiers:
        data.append(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
    data = np.array(data, dtype=np.float64)
    
    # 执行 Nemenyi 事后检验
    nemenyi = posthoc_nemenyi_friedman(data.T)
    
    # 将 p 值结果整理为列表
    p_values = []
    for nemenyi_indx in nemenyi.index:
        for nemenyi_columns in nemenyi.columns:
            if nemenyi_indx < nemenyi_columns:
                if nemenyi.loc[nemenyi_indx, nemenyi_columns] < alpha:
                    p_values.append((classifiers[nemenyi_indx], classifiers[nemenyi_columns], 
                                     nemenyi.loc[nemenyi_indx, nemenyi_columns], True))
                else:
                    p_values.append((classifiers[nemenyi_indx], classifiers[nemenyi_columns], 
                                     nemenyi.loc[nemenyi_indx, nemenyi_columns], False))
    
    # 计算每个分类器的平均排名
    m = len(classifiers)
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)]. \
        sort_values(['classifier_name', 'dataset_name'])
    
    rank_data = np.array(sorted_df_perf['accuracy']).reshape(m, max_nb_datasets)
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), 
                           columns=np.unique(sorted_df_perf['dataset_name']))
    
    # 计算平均排名
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    
    return p_values, average_ranks, max_nb_datasets

def form_cliques(p_values, nnames):
    """根据统计显著性测试结果形成算法组（无显著差异的算法组）"""
    import networkx as nx
    
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    
    # 构建无向图：如果两个算法之间没有显著差异，添加边
    for p in p_values:
        if p[3] == False:  # p[3]=False 表示无显著差异
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1
    
    g = nx.Graph(g_data)
    
    # 返回无向图中的所有最大团（最大完全子图）
    # 团表示一组算法，它们之间两两没有显著差异
    return nx.find_cliques(g)

def prepare_benchmark_data(df, algorithm_list, filter_condition=None, metric_column=None):
    """准备基准测试数据，用于统计分析
    
    参数:
        df: 基准测试数据DataFrame
        algorithm_list: 要分析的算法列表
        filter_condition: 可选的过滤条件
        metric_column: 指定的指标列名 (如'AUC-PR', 'VUS-PR'等)
    
    返回:
        包含分类器名称、数据集名称和准确率的DataFrame
    """
    if filter_condition is not None:
        df_filter = filter_condition(df)
    else:
        df_filter = df
        
    eval_list = []
    # 检查DataFrame结构
    is_new_format = 'Algorithm' in df.columns and 'Dataset' in df.columns
    
    if is_new_format:
        # 新格式: 每行是一个算法在一个数据集上的结果
        for _, row in df_filter.iterrows():
            algo = row['Algorithm']
            if algo in algorithm_list:
                if metric_column is not None and metric_column in row:
                    # 转换为数值，非数值则跳过
                    try:
                        metric_value = float(row[metric_column])
                        eval_list.append([algo, row['Dataset'], metric_value])
                    except (ValueError, TypeError):
                        continue
    else:
        # 旧格式: 列是算法，行是数据集
        for index, row in df_filter.iterrows():
            for method in algorithm_list:
                if method in row and pd.notna(row[method]):  # 确保方法存在且值不为 NaN
                    dataset_name = row['file'] if 'file' in row else f"dataset_{index}"
                    # 如果指定了指标列，则使用该列的值
                    metric_value = row[method]
                    eval_list.append([method, dataset_name, metric_value])
    
    eval_df = pd.DataFrame(eval_list, columns=['classifier_name', 'dataset_name', 'accuracy'])
    return eval_df 