# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import torch
import random, argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, f1_score
from .evaluation.metrics import get_metrics
from .utils.slidingWindows import find_length_rank
from .model_wrapper import *
from .HP_list import Optimal_Uni_algo_HP_dict

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA Available: ", torch.cuda.is_available())
print("cuDNN Version: ", torch.backends.cudnn.version())

def get_threshold_by_roc_curve(output, label):
    """使用ROC曲线找到最佳阈值"""
    fpr, tpr, thresholds = roc_curve(label, output)
    # 使用Youden's J统计量找到最佳阈值
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def get_threshold_by_f1_search(output, label, step_size=0.001):
    """通过网格搜索找到最佳F1分数对应的阈值"""
    best_f1 = 0
    best_threshold = 0
    
    for threshold in np.arange(0, 1, step_size):
        pred = output > threshold
        f1 = f1_score(label, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    return best_threshold

if __name__ == '__main__':

    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running TSB-AD')
    # parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    # parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    # 要求的uni数据
    parser.add_argument('--filename', type=str, default='451_UCR_id_149_Medical_tr_3000_1st_7175.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    # 要求的multi数据
    # parser.add_argument('--filename', type=str, default='178_Exathlon_id_5_Facility_tr_12538_1st_12638.csv')
    # parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-M/')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--AD_Name', type=str, default='IForest')
    args = parser.parse_args()

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    # 使用滑动窗口捕捉时间序列中的局部模式
    slidingWindow = find_length_rank(data, rank=1)
    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]
    Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]

    if args.AD_Name in Semisupervise_AD_Pool:
        output = run_Semisupervise_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)
    elif args.AD_Name in Unsupervise_AD_Pool:
        output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
    else:
        raise Exception(f"{args.AD_Name} is not defined")

    if isinstance(output, np.ndarray):
        # 对数据归一化
        output = MinMaxScaler(feature_range=(0,1)).fit_transform(output.reshape(-1,1)).ravel()
        
        # 方法1：使用ROC曲线找到最佳阈值
        threshold_roc = get_threshold_by_roc_curve(output, label)
        pred_roc = output > threshold_roc
        
        # 方法2：通过网格搜索找到最佳F1分数对应的阈值
        threshold_f1 = get_threshold_by_f1_search(output, label)
        pred_f1 = output > threshold_f1
        
        # 方法3：原始方法（均值+3倍标准差）
        threshold_std = np.mean(output) + 3 * np.std(output)
        pred_std = output > threshold_std
        
        # 计算三种方法的评估结果
        evaluation_result_roc = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred_roc)
        evaluation_result_f1 = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred_f1)
        evaluation_result_std = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred_std)
        
        print('Evaluation Result (ROC threshold): ', evaluation_result_roc)
        print('Evaluation Result (F1 threshold): ', evaluation_result_f1)
        print('Evaluation Result (Mean+3*Std threshold): ', evaluation_result_std)
        
        # 打印阈值
        print(f'ROC threshold: {threshold_roc:.4f}')
        print(f'F1 threshold: {threshold_f1:.4f}')
        print(f'Mean+3*Std threshold: {threshold_std:.4f}')
    else:
        print(f'At {args.filename}: '+output)

