# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, f1_score

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

class Custom_AD(BaseDetector):

    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        n_samples, n_features = X.shape
        if self.normalize: X = zscore(X, axis=1, ddof=1)

        self.decision_scores_ = np.zeros(n_samples)
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        使用拟合的检测器预测X的原始异常分数。
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        输入样本的异常分数是根据不同的检测器算法; 
        为了保持一致性, 异常值被分配为更大的异常分数

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        n_samples, n_features = X.shape
        decision_scores_ = np.zeros(n_samples)
        return decision_scores_


def run_Custom_AD_Unsupervised(data, HP):
    '''
    无监督学习函数
    '''
    clf = Custom_AD(HP=HP)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_Custom_AD_Semisupervised(data_train, data_test, HP):
    
    '''
    半监督学习函数
    '''
    clf = Custom_AD(HP=HP)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

# 新增的
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
    Start_T = time.time()
    
    # 设置随机种子
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

    # 参数设置
    parser = argparse.ArgumentParser(description='Running Custom_AD')
    parser.add_argument('--dataset_type', type=str, default='uni', choices=['uni', 'multi'], 
                       help='选择数据集类型：uni(单变量)或multi(多变量)')
    parser.add_argument('--filename', type=str, 
                       default='451_UCR_id_149_Medical_tr_3000_1st_7175.csv',  # 单变量数据
                       help='数据文件名')
    parser.add_argument('--data_direc', type=str, 
                       default='../Datasets/TSB-AD-U/',  # 单变量数据目录
                       help='数据目录路径')
    parser.add_argument('--save', type=bool, default=False, 
                       help='是否保存评估结果')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/custom/', 
                       help='评估结果保存目录')
    args = parser.parse_args()

    # 根据数据集类型设置默认值
    if args.dataset_type == 'multi':
        args.filename = '178_Exathlon_id_5_Facility_tr_12538_1st_12638.csv'
        args.data_direc = '../Datasets/TSB-AD-M/'

    # 创建保存目录
    if args.save:
        os.makedirs(args.save_dir, exist_ok=True)

    # 读取数据
    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data shape: ', data.shape)
    print('label shape: ', label.shape)

    # 设置超参数
    Custom_AD_HP = {
        'window_size': 10,
        'hidden_dim': 64,
        'threshold_factor': 3,
    }

    # 计算滑动窗口
    if args.dataset_type == 'uni':
        slidingWindow = find_length_rank(data, rank=1)
    else:
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)

    # 分割训练集和测试集
    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]

    # 运行模型
    start_time = time.time()
    output = run_Custom_AD_Semisupervised(data_train, data, **Custom_AD_HP)
    end_time = time.time()
    run_time = end_time - start_time

    # 计算三种阈值
    threshold_roc = get_threshold_by_roc_curve(output, label)
    threshold_f1 = get_threshold_by_f1_search(output, label)
    threshold_std = np.mean(output) + 3 * np.std(output)

    # 生成预测结果
    pred_roc = output > threshold_roc
    pred_f1 = output > threshold_f1
    pred_std = output > threshold_std

    # 计算评估指标
    evaluation_result_roc = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred_roc)
    evaluation_result_f1 = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred_f1)
    evaluation_result_std = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred_std)

    # 打印结果
    print(f'\nDataset: {args.filename}')
    print(f'Dataset Type: {args.dataset_type}')
    print(f'Run Time: {run_time:.3f}s')
    
    print('\nEvaluation Results:')
    print('ROC threshold:')
    print(f'  Threshold: {threshold_roc:.4f}')
    print(f'  Metrics: {evaluation_result_roc}')
    
    print('\nF1 threshold:')
    print(f'  Threshold: {threshold_f1:.4f}')
    print(f'  Metrics: {evaluation_result_f1}')
    
    print('\nMean+3*Std threshold:')
    print(f'  Threshold: {threshold_std:.4f}')
    print(f'  Metrics: {evaluation_result_std}')

    # 保存结果
    if args.save:
        results = {
            'filename': args.filename,
            'dataset_type': args.dataset_type,
            'run_time': run_time,
            'roc_threshold': threshold_roc,
            'f1_threshold': threshold_f1,
            'std_threshold': threshold_std,
            'roc_metrics': evaluation_result_roc,
            'f1_metrics': evaluation_result_f1,
            'std_metrics': evaluation_result_std
        }
        
        # 保存为CSV
        df_results = pd.DataFrame([results])
        df_results.to_csv(f'{args.save_dir}/custom_results.csv', index=False)