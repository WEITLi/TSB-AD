# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
# 注意：不再需要导入 nn, PCA, BaseDetector, zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score, roc_auc_score
# 从 TSB_AD 导入所需模块
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD, run_TCN # 直接导入 run_TCN 函数
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict # 导入最优超参数
import random # 导入 random 以便设置种子
import os # 导入 os 模块用于路径操作
import inspect # 导入 inspect 模块

# --- Debug: 打印 run_Semisupervise_AD 的来源 ---
print(f"--- Debug: run_Semisupervise_AD imported from: {inspect.getfile(run_Semisupervise_AD)}")
print(f"--- Debug: run_Unsupervise_AD imported from: {inspect.getfile(run_Unsupervise_AD)}")
try: # 添加 try-except 来检查 run_TCN 是否成功导入
    print(f"--- Debug: run_TCN imported from: {inspect.getfile(run_TCN)}")
except Exception as e:
    print(f"--- Debug: run_TCN 导入失败: {e}")
# ---------------------------------------------

# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建到项目根目录 (TSB-AD/TSB-AD/) 的路径
project_root = os.path.abspath(os.path.join(script_dir, '..')) 
# 构建到 Datasets 目录的基路径
datasets_base_path = os.path.join(project_root, 'Datasets')

def get_threshold_by_roc_curve(output, label):
    """使用ROC曲线找到最佳阈值"""
    # 确保 output 和 label 是 numpy 数组
    output = np.asarray(output)
    label = np.asarray(label)
    # 1. 计算 ROC 曲线的关键点
    #    output: 模型输出的异常分数 (连续值)
    #    label: 真实的标签 (0 或 1)
    #    返回: fpr (假正例率), tpr (真正例率=召回率), thresholds (对应的阈值)
    fpr, tpr, thresholds = roc_curve(label, output) 
    
    # 处理 NaN 值 (例如当所有预测分数相同时可能出现)
    valid_indices = ~np.isnan(tpr) & ~np.isnan(fpr)
    if not np.any(valid_indices):
        return 0.5 # 或者其他默认值

    fpr, tpr, thresholds = fpr[valid_indices], tpr[valid_indices], thresholds[valid_indices]
    if len(thresholds) == 0:
        return 0.5 # 如果没有有效阈值，返回默认值

    # 2. 找到 "最佳" 阈值对应的索引
    #    计算 tpr - fpr (也称为 Youden's J statistic)
    #    找到这个差值最大的点，表示该点距离随机猜测线(y=x)最远，
    #    是 TPR 很高同时 FPR 很低的一个较好的平衡点。
    optimal_idx = np.argmax(tpr - fpr) 

    # 3. 获取该索引对应的阈值
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold # 返回找到的最优阈值

def get_threshold_by_f1_search(output, label, step_size=0.001):
    """通过网格搜索找到最佳F1分数对应的阈值"""
    # 确保 output 和 label 是 numpy 数组
    output = np.asarray(output)
    label = np.asarray(label)
    best_f1 = -1 # 初始化为-1以确保任何计算出的f1都会更高
    best_threshold = 0
    
    min_val, max_val = np.min(output), np.max(output)
    if min_val == max_val: # 如果所有输出值都相同，则无法进行有意义的搜索
         return 0.5 # 或其他默认值

    # 生成阈值，覆盖输出范围
    thresholds = np.arange(min_val, max_val, (max_val - min_val) * step_size)
    if len(thresholds) == 0:
        thresholds = np.array([np.mean(output)]) # 如果步长太大，至少尝试一个阈值

    for threshold in thresholds:
        pred = (output >= threshold).astype(int)
        # 检查 label 是否只有一个类别
        if len(np.unique(label)) < 2:
             # 如果只有一个类别，F1分数未定义或无意义
             # 根据情况决定如何处理，例如跳过或返回默认阈值
             continue # 或者 return 0.5
        f1 = f1_score(label, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    # 如果从未找到比-1更好的F1，返回默认值
    if best_f1 == -1:
        return 0.5

    return best_threshold

def evaluate_metrics(output, label, threshold):
    """计算评估指标"""
    # 确保 output 和 label 是 numpy 数组
    output = np.asarray(output)
    label = np.asarray(label)
    threshold = float(threshold) # 确保阈值是浮点数

    # 检查 label 是否只有一个类别
    if len(np.unique(label)) < 2:
        print("警告: 标签只包含一个类别，评估指标可能无意义。")
        # 返回默认值或根据需要处理
        return {
            'precision': 0.0,
            'recall': 0.0,
            'roc_auc': 0.5
        }
    
    pred = (output >= threshold).astype(int)
    precision = precision_score(label, pred, zero_division=0)
    recall = recall_score(label, pred, zero_division=0)
    try:
        # ROC AUC 应该总是基于原始分数计算，而不是预测标签
        roc_auc = roc_auc_score(label, output) 
    except ValueError as e:
         # 当标签中只有一个类别时，roc_auc_score会报错
         print(f"计算 ROC AUC 时出错: {e}。返回 0.5 作为默认值。")
         roc_auc = 0.5

    return {
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc # 注意：对于给定的阈值，ROC_AUC是一样的，但在此处返回以便打印
    }

if __name__ == '__main__':
    # 设置随机种子
    seed = 2024
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 单变量数据示例
    print("处理单变量数据 (使用 Sub_PCA)...")
    # 使用 os.path.join 构建绝对路径
    uni_filename = '451_UCR_id_149_Medical_tr_3000_1st_7175.csv'
    uni_data_path = os.path.join(datasets_base_path, 'TSB-AD-U', uni_filename) 
    print(f"尝试加载单变量数据: {uni_data_path}")
    try:
        df_uni = pd.read_csv(uni_data_path).dropna()
    except FileNotFoundError:
        print(f"错误: 文件未找到于 {uni_data_path}")
        print("请确认 'Datasets/TSB-AD-U' 目录相对于 TSB-AD/TSB-AD/ 存在且包含所需文件。")
        exit() # 或者进行其他错误处理

    data_uni = df_uni.iloc[:, 0:-1].values.astype(float)
    label_uni = df_uni['Label'].astype(int).to_numpy()
    
    # 使用 model_wrapper 调用 Sub_PCA
    AD_Name_uni = 'Sub_PCA'
    # 从 HP_list 获取 Sub_PCA 的最优超参数
    # 注意：Sub_PCA 的 HP 可能需要 'periodicity'，但 model_wrapper 中的 run_Sub_PCA 会处理
    # 如果 Optimal_Uni_algo_HP_dict['Sub_PCA'] 为空或不包含所需参数，则使用默认值
    hp_uni = Optimal_Uni_algo_HP_dict.get(AD_Name_uni, {}) 
    print(f"使用超参数: {hp_uni}")
    output_uni = run_Unsupervise_AD(AD_Name_uni, data_uni, **hp_uni)
    
    # 归一化输出分数
    if isinstance(output_uni, np.ndarray) and output_uni.size > 0:
        if not np.all(np.isnan(output_uni)) and np.ptp(output_uni[~np.isnan(output_uni)]) > 0:
             output_uni = MinMaxScaler(feature_range=(0,1)).fit_transform(output_uni.reshape(-1,1)).ravel()
        else:
            print("警告: Sub_PCA 输出无效或恒定，跳过归一化和评估。")
            output_uni = None # 将 output_uni 设为 None 以跳过评估

        # --- 修改评估部分 --- 
        if output_uni is not None and output_uni.size > 0 and len(np.unique(label_uni)) >= 2:
            # 计算两种阈值
            threshold_uni_roc = get_threshold_by_roc_curve(output_uni, label_uni)
            threshold_uni_f1 = get_threshold_by_f1_search(output_uni, label_uni)
            
            # 使用两种阈值分别评估
            metrics_uni_roc = evaluate_metrics(output_uni, label_uni, threshold_uni_roc)
            metrics_uni_f1 = evaluate_metrics(output_uni, label_uni, threshold_uni_f1)
            
            # 打印两种评估结果
            print("\n单变量数据评估结果 (基于 ROC 阈值):")
            print(f"  阈值: {threshold_uni_roc:.4f}")
            print(f"  Precision: {metrics_uni_roc['precision']:.4f}")
            print(f"  Recall: {metrics_uni_roc['recall']:.4f}")
            print(f"  ROC-AUC: {metrics_uni_roc['roc_auc']:.4f}")

            print("\n单变量数据评估结果 (基于最佳 F1 阈值):")
            print(f"  阈值: {threshold_uni_f1:.4f}")
            print(f"  Precision: {metrics_uni_f1['precision']:.4f}")
            print(f"  Recall: {metrics_uni_f1['recall']:.4f}")
            print(f"  ROC-AUC: {metrics_uni_f1['roc_auc']:.4f}") # ROC AUC 与阈值无关，但保持打印格式一致
        elif output_uni is None:
             print("\n单变量数据评估跳过，因为输出分数无效或恒定。")
        else: # len(np.unique(label_uni)) < 2
             print("\n单变量数据评估跳过，因为标签只包含一个类别。")
        # --- 评估部分修改结束 --- 

    else:
        print(f"运行 {AD_Name_uni} 失败或返回无效输出: {output_uni}")
    
    # 多变量数据示例
    print("\n处理多变量数据 (使用 DTAAD)...")
    # 使用 os.path.join 构建绝对路径
    multi_filename = '178_Exathlon_id_5_Facility_tr_12538_1st_12638.csv'
    multi_data_path = os.path.join(datasets_base_path, 'TSB-AD-M', multi_filename)
    print(f"尝试加载多变量数据: {multi_data_path}")
    try:
        df_multi = pd.read_csv(multi_data_path).dropna()
    except FileNotFoundError:
        print(f"错误: 文件未找到于 {multi_data_path}")
        print("请确认 'Datasets/TSB-AD-M' 目录相对于 TSB-AD/TSB-AD/ 存在且包含所需文件。")
        exit() # 或者进行其他错误处理

    data_multi = df_multi.iloc[:, 0:-1].values.astype(float)
    label_multi = df_multi['Label'].astype(int).to_numpy()
    
    # 分割训练集和测试集
    # 确保文件名中包含训练索引信息，格式如 '..._tr_INDEX_...' 或 '..._INDEX_...' 
    try:
        # 优先使用 multi_filename 进行分割
        base_name = multi_filename.split('.')[0]
        parts = base_name.split('_')
        # 查找 'tr' 后面的数字或者最后一个数字作为索引
        train_index_str = None
        if 'tr' in parts:
            try:
                tr_index = parts.index('tr')
                if tr_index + 1 < len(parts) and parts[tr_index + 1].isdigit():
                    train_index_str = parts[tr_index + 1]
            except ValueError:
                pass # 'tr' not found
        
        if train_index_str is None: # 如果 'tr' 模式未找到，尝试最后一个数字
             train_index_str = next((p for p in reversed(parts) if p.isdigit()), None)

        if train_index_str is None:
            raise ValueError("无法从文件名中提取训练索引")
            
        train_index = int(train_index_str)
        if train_index <= 0 or train_index >= len(data_multi): # 增加索引有效性检查
             raise ValueError(f"从文件名解析的训练索引 {train_index} 无效 (数据长度 {len(data_multi)})")
        data_train = data_multi[:train_index, :]
        print(f"从文件名解析得到的训练集大小: {train_index}")

    except Exception as e:
        print(f"警告：无法从文件名 '{multi_filename}' 自动提取训练索引: {e}")
        print("将使用前 80% 的数据作为训练集。")
        train_index = int(len(data_multi) * 0.8)
        data_train = data_multi[:train_index, :]

    # 使用 model_wrapper 调用 DTAAD
    AD_Name_multi = 'DTAAD'
    # 从 HP_list 获取 DTAAD 的参数，如果没有则使用空字典（使用默认参数）
    hp_multi = Optimal_Multi_algo_HP_dict.get(AD_Name_multi, {}) 
    print(f"使用超参数: {hp_multi}")
    # DTAAD 是半监督的，需要训练和测试数据
    
    # 尝试通过 run_Semisupervise_AD 间接调用 run_DTAAD
    print("尝试通过 run_Semisupervise_AD 间接调用 DTAAD 模型")
    output_multi = run_Semisupervise_AD(AD_Name_multi, data_train, data_multi, **hp_multi)
    
    # 归一化输出分数
    if isinstance(output_multi, np.ndarray) and output_multi.size > 0:
        if not np.all(np.isnan(output_multi)) and np.ptp(output_multi[~np.isnan(output_multi)]) > 0:
            output_multi = MinMaxScaler(feature_range=(0,1)).fit_transform(output_multi.reshape(-1,1)).ravel()
        else:
            print("警告: DTAAD 输出无效或恒定，跳过归一化和评估。")
            output_multi = None # 将 output_multi 设为 None 以跳过评估

        # --- 修改评估部分 --- 
        if output_multi is not None and output_multi.size > 0 and len(np.unique(label_multi)) >= 2:
            # 计算两种阈值
            threshold_multi_roc = get_threshold_by_roc_curve(output_multi, label_multi)
            threshold_multi_f1 = get_threshold_by_f1_search(output_multi, label_multi)

            # 使用两种阈值分别评估
            metrics_multi_roc = evaluate_metrics(output_multi, label_multi, threshold_multi_roc)
            metrics_multi_f1 = evaluate_metrics(output_multi, label_multi, threshold_multi_f1)

            # 打印两种评估结果
            print("\n多变量数据评估结果 (基于 ROC 阈值):")
            print(f"  阈值: {threshold_multi_roc:.4f}")
            print(f"  Precision: {metrics_multi_roc['precision']:.4f}")
            print(f"  Recall: {metrics_multi_roc['recall']:.4f}")
            print(f"  ROC-AUC: {metrics_multi_roc['roc_auc']:.4f}")

            print("\n多变量数据评估结果 (基于最佳 F1 阈值):")
            print(f"  阈值: {threshold_multi_f1:.4f}")
            print(f"  Precision: {metrics_multi_f1['precision']:.4f}")
            print(f"  Recall: {metrics_multi_f1['recall']:.4f}")
            print(f"  ROC-AUC: {metrics_multi_f1['roc_auc']:.4f}")
        elif output_multi is None:
             print("\n多变量数据评估跳过，因为输出分数无效或恒定。")
        else: # len(np.unique(label_multi)) < 2
             print("\n多变量数据评估跳过，因为标签只包含一个类别。")
        # --- 评估部分修改结束 --- 

    else:
        print(f"运行 {AD_Name_multi} 失败或返回无效输出: {output_multi}")
