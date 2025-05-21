import pandas as pd
import numpy as np
import torch
import random
import argparse
import time
import os
import sys
import wandb
import json
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize

# 添加项目根目录到系统路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Multi_algo_HP_dict

# 设置随机种子
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class ExperimentManager:
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{args.AD_Name}_{self.timestamp}"
        
        # 创建实验目录
        self.exp_dir = os.path.join(args.save_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 保存实验配置
        self.save_config()
        
    def save_config(self):
        """保存实验配置"""
        config = vars(self.args)
        config['timestamp'] = self.timestamp
        config['seed'] = seed
        
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
            
    def save_checkpoint(self, iteration, optimizer, results):
        """保存检查点"""
        checkpoint = {
            'iteration': iteration,
            'optimizer_state': {
                'X': optimizer.X,
                'y': optimizer.y
            },
            'results': results
        }
        
        checkpoint_path = os.path.join(self.exp_dir, f'checkpoint_{iteration}.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=4)
            
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint

class BayesianOptimizer:
    def __init__(self, param_bounds, n_initial_points=5):
        self.param_bounds = param_bounds
        self.n_initial_points = n_initial_points
        self.X = []  # 参数组合
        self.y = []  # 目标值
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
    def _random_sample(self):
        """随机采样一个参数组合"""
        params = {}
        for param_name, (min_val, max_val) in self.param_bounds.items():
            if param_name in ['window_size', 'batch_size', 'epochs', 'pred_len']:
                # 整数参数
                params[param_name] = int(np.random.randint(int(min_val), int(max_val) + 1))
            else:
                # 浮点数参数
                params[param_name] = float(np.random.uniform(float(min_val), float(max_val)))
        return params
    
    def _acquisition_function(self, x, xi=0.01):
        """计算采集函数 (Expected Improvement)"""
        mean, std = self.gp.predict(x.reshape(1, -1), return_std=True)
        best_f = np.max(self.y) if self.y else 0
        
        z = (mean - best_f - xi) / std
        ei = (mean - best_f - xi) * norm.cdf(z) + std * norm.pdf(z)
        return -ei  # 最小化负的EI
    
    def suggest(self):
        """建议下一个要评估的参数组合"""
        if len(self.X) < self.n_initial_points:
            return self._random_sample()
        
        # 将参数转换为数值数组
        X_array = np.array([self._param_to_array(p) for p in self.X])
        y_array = np.array(self.y)
        
        # 更新高斯过程
        self.gp.fit(X_array, y_array)
        
        # 优化采集函数
        result = minimize(
            self._acquisition_function,
            x0=self._param_to_array(self._random_sample()),
            bounds=[self.param_bounds[p] for p in self.param_bounds],
            method='L-BFGS-B'
        )
        
        # 将结果转换回参数字典
        return self._array_to_param(result.x)
    
    def _param_to_array(self, params):
        """将参数字典转换为数值数组"""
        return np.array([params[p] for p in self.param_bounds])
    
    def _array_to_param(self, array):
        """将数值数组转换为参数字典"""
        params = {}
        for i, (param_name, (min_val, max_val)) in enumerate(self.param_bounds.items()):
            if param_name in ['window_size', 'batch_size', 'epochs', 'pred_len']:
                # 整数参数
                params[param_name] = int(array[i])
            else:
                # 浮点数参数
                params[param_name] = float(array[i])
        return params
    
    def update(self, params, score):
        """更新优化器的观察结果"""
        self.X.append(params)
        self.y.append(score)

def run_bayesian_optimization(args):
    # 初始化实验管理器
    exp_manager = ExperimentManager(args)
    
    # 初始化wandb
    wandb.init(
        project="TSB-AD-HP-Tuning",
        name=exp_manager.exp_name,
        config={
            "algorithm": args.AD_Name,
            "optimization_target": "VUS-PR",
            "n_iterations": args.n_iterations,
            "n_initial_points": args.n_initial_points,
            "timestamp": exp_manager.timestamp
        }
    )
    
    try:
        # 加载数据集列表
        file_list_path = os.path.abspath(os.path.join(script_dir, args.file_list))
        print(f"正在读取数据集列表: {file_list_path}")
        
        try:
            file_list = pd.read_csv(file_list_path)['file_name'].values
            print(f"找到 {len(file_list)} 个数据集")
        except Exception as e:
            print(f"读取数据集列表失败: {str(e)}")
            raise
            
        if args.dataset:
            file_list = [args.dataset]
            print(f"使用单个数据集: {args.dataset}")
            
        # 过滤出存在的文件
        existing_files = []
        for file_name in file_list:
            file_path = os.path.join(args.dataset_dir, file_name)
            if os.path.exists(file_path):
                existing_files.append(file_name)
                print(f"找到数据集: {file_name}")
            else:
                print(f"警告: 数据集 {file_name} 不存在,已跳过")
        
        if not existing_files:
            raise ValueError("没有找到可用的数据集文件")
            
        file_list = existing_files
        print(f"将使用 {len(file_list)} 个数据集进行优化")
        
        # 获取超参数搜索空间
        param_space = Multi_algo_HP_dict[args.AD_Name]
        
        # 定义参数边界
        param_bounds = {
            'window_size': (50, 200),  # 整数
            'lr': (1e-4, 5e-3),        # 浮点数
            'batch_size': (32, 128),   # 整数，减小最大值
            'epochs': (30, 200),       # 整数
            'pred_len': (1, 5),        # 整数
            'validation_size': (0.1, 0.3)  # 浮点数
        }
        
        # 初始化贝叶斯优化器
        optimizer = BayesianOptimizer(param_bounds, n_initial_points=args.n_initial_points)
        
        # 存储结果
        results = []
        
        # 运行优化
        for i in range(args.n_iterations):
            print(f"\n迭代 {i+1}/{args.n_iterations}")
            
            # 获取建议的参数
            params = optimizer.suggest()
            print(f"建议的参数: {params}")
            
            # 在数据集上评估参数
            dataset_scores = []
            dataset_metrics = []
            
            for filename in file_list:
                print(f"处理数据集: {filename}")
                
                try:
                    # 加载数据
                    file_path = os.path.join(args.dataset_dir, filename)
                    df = pd.read_csv(file_path).dropna()
                    data = df.iloc[:, 0:-1].values.astype(float)
                    label = df['Label'].astype(int).to_numpy()
                    
                    # 准备训练数据
                    feats = data.shape[1]
                    slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
                    train_index = filename.split('.')[0].split('_')[-3]
                    data_train = data[:int(train_index), :]
                    
                    # 运行模型
                    if args.AD_Name in Semisupervise_AD_Pool:
                        output = run_Semisupervise_AD(args.AD_Name, data_train, data, **params)
                    else:
                        output = run_Unsupervise_AD(args.AD_Name, data, **params)
                    
                    # 计算评估指标
                    metrics = get_metrics(output, label, slidingWindow=slidingWindow)
                    score = metrics['VUS-PR']  # 使用VUS-PR作为优化目标
                    dataset_scores.append(score)
                    dataset_metrics.append(metrics)
                    
                except Exception as e:
                    print(f"处理数据集 {filename} 时出错: {str(e)}")
                    dataset_scores.append(0)
                    dataset_metrics.append({})
            
            # 计算平均分数
            avg_score = np.mean(dataset_scores)
            print(f"平均VUS-PR分数: {avg_score}")
            
            # 更新优化器
            optimizer.update(params, avg_score)
            
            # 记录到wandb
            wandb.log({
                "iteration": i + 1,
                "avg_vus_pr": avg_score,
                **params,
                "best_score_so_far": max(optimizer.y)
            })
            
            # 记录每个数据集的详细指标
            for j, (filename, metrics) in enumerate(zip(file_list, dataset_metrics)):
                if metrics:
                    wandb.log({
                        f"dataset_{j}_metrics": metrics,
                        "iteration": i + 1,
                        "dataset": filename
                    })
            
            # 保存结果
            results.append({
                'iteration': i + 1,
                'params': params,
                'score': avg_score
            })
            
            # 保存检查点
            exp_manager.save_checkpoint(i + 1, optimizer, results)
            
            # 保存中间结果
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(exp_manager.exp_dir, f'{args.AD_Name}_bayesian_results.csv'), index=False)
        
        # 找到最佳参数
        best_idx = np.argmax([r['score'] for r in results])
        best_params = results[best_idx]['params']
        best_score = results[best_idx]['score']
        
        print(f"\n最佳参数: {best_params}")
        print(f"最佳VUS-PR分数: {best_score}")
        
        # 记录最终结果到wandb
        wandb.log({
            "best_params": best_params,
            "best_score": best_score
        })
        
        return best_params, best_score
        
    except Exception as e:
        print(f"优化过程出错: {str(e)}")
        raise
        
    finally:
        # 关闭wandb
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian Hyperparameter Optimization')
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/TSB-AD-M/')
    parser.add_argument('--file_list', type=str, default='../Datasets/File_List/TSB-AD-M-Tuning.csv')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='eval/HP_tuning/bayesian/')
    parser.add_argument('--AD_Name', type=str, default='DLinear')
    parser.add_argument('--n_iterations', type=int, default=20)
    parser.add_argument('--n_initial_points', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复实验')
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 运行贝叶斯优化
    best_params, best_score = run_bayesian_optimization(args) 