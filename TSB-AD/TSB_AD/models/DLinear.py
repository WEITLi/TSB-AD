import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ForecastDataset

class DLinearModel(nn.Module):
    def __init__(self, window_size, feats, pred_len, device):
        super().__init__()
        self.pred_len = pred_len
        self.feats = feats
        self.device = device
        
        # 趋势分解
        self.trend_linear = nn.Linear(window_size, pred_len)
        # 季节性分解
        self.seasonal_linear = nn.Linear(window_size, pred_len)
        # 最终预测层
        self.final_linear = nn.Linear(feats * 2, feats)
        
    def forward(self, x):
        # x shape: (batch_size, window_size, feats)
        batch_size = x.shape[0]
        
        # 趋势分解
        trend = self.trend_linear(x.transpose(1, 2))  # (batch_size, feats, pred_len)
        
        # 季节性分解
        seasonal = self.seasonal_linear(x.transpose(1, 2))  # (batch_size, feats, pred_len)
        
        # 合并趋势和季节性
        combined = torch.cat([trend, seasonal], dim=1)  # (batch_size, feats*2, pred_len)
        
        # 最终预测
        output = self.final_linear(combined.transpose(1, 2))  # (batch_size, pred_len, feats)
        
        return output

class DLinear():
    def __init__(self,
                 window_size=100,
                 pred_len=1,
                 batch_size=128,
                 epochs=50,
                 lr=0.001,
                 feats=1,
                 validation_size=0.2):
        super().__init__()
        self.__anomaly_score = None
        
        cuda = True
        self.device = get_gpu(cuda)
        
        self.window_size = window_size
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.lr = lr
        self.validation_size = validation_size
        
        print(f"初始化 DLinear 模型:")
        print(f"- 窗口大小: {window_size}")
        print(f"- 特征数: {feats}")
        print(f"- 预测长度: {pred_len}")
        print(f"- 批次大小: {batch_size}")
        
        self.model = DLinearModel(self.window_size, feats, self.pred_len, self.device).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.early_stopping = EarlyStoppingTorch(patience=3)
        
        self.mu = None
        self.sigma = None
        self.eps = 1e-10
        
    def fit(self, data):
        print(f"\n开始训练:")
        print(f"输入数据维度: {data.shape}")
        
        # 确保数据是浮点型
        data = data.astype(np.float32)
        
        # 计算实际batch size
        actual_batch_size = min(self.batch_size, len(data) // 4)
        if actual_batch_size < 1:
            actual_batch_size = 1
        print(f"实际使用的batch size: {actual_batch_size}")
        
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]
        
        train_loader = DataLoader(
            ForecastDataset(tsTrain, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=actual_batch_size,
            shuffle=True,
            num_workers=0  # 避免多进程
        )
        
        valid_loader = DataLoader(
            ForecastDataset(tsValid, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=actual_batch_size,
            shuffle=False,
            num_workers=0  # 避免多进程
        )
        
        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            self.model.train()
            train_loss = 0
            for batch_idx, (x, target) in enumerate(train_loader):
                print(f"\n训练批次 {batch_idx + 1}:")
                print(f"输入 X shape: {x.shape}")
                
                x, target = x.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x)
                print(f"模型输出 shape: {output.shape}")
                
                # 调整输出和目标维度
                output = output.view(-1, self.feats)
                target = target.view(-1, self.feats)
                print(f"调整后的输出 shape: {output.shape}")
                print(f"调整后的目标 shape: {target.shape}")
                
                # 计算损失
                loss = self.loss(output, target)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
                
                print(f"批次损失: {loss.item():.6f}")
            
            self.model.eval()
            valid_loss = 0
            scores = []
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(valid_loader):
                    print(f"\n验证批次 {batch_idx + 1}:")
                    print(f"输入 X shape: {x.shape}")
                    
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    print(f"模型输出 shape: {output.shape}")
                    
                    # 调整输出和目标维度
                    output = output.view(-1, self.feats)
                    target = target.view(-1, self.feats)
                    print(f"调整后的输出 shape: {output.shape}")
                    print(f"调整后的目标 shape: {target.shape}")
                    
                    # 计算损失
                    loss = self.loss(output, target)
                    valid_loss += loss.item()
                    
                    # 计算异常分数
                    mse = torch.sub(output, target).pow(2)
                    scores.append(mse.cpu())
                    print(f"批次损失: {loss.item():.6f}")
            
            valid_loss /= len(valid_loader)
            self.scheduler.step()
            
            print(f"\nEpoch {epoch} 总结:")
            print(f"训练损失: {train_loss/len(train_loader):.6f}")
            print(f"验证损失: {valid_loss:.6f}")
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            if len(scores) > 0:
                scores = torch.cat(scores, dim=0)
                self.mu = torch.mean(scores)
                self.sigma = torch.var(scores)
                print(f"异常分数统计:")
                print(f"- 均值: {self.mu:.6f}")
                print(f"- 方差: {self.sigma:.6f}")
    
    def decision_function(self, data):
        print(f"\n开始预测:")
        print(f"输入数据维度: {data.shape}")
        
        try:
            # 确保数据是浮点型
            data = data.astype(np.float32)
            
            # 计算实际batch size
            actual_batch_size = min(self.batch_size, len(data) // 4)
            if actual_batch_size < 1:
                actual_batch_size = 1
            print(f"实际使用的batch size: {actual_batch_size}")
            
            test_loader = DataLoader(
                ForecastDataset(data, window_size=self.window_size, pred_len=self.pred_len),
                batch_size=actual_batch_size,
                shuffle=False,
                num_workers=0  # 避免多进程
            )
            
            self.model.eval()
            scores = []
            outputs = []
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(test_loader):
                    print(f"\n预测批次 {batch_idx + 1}:")
                    print(f"输入 X shape: {x.shape}")
                    
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    print(f"模型输出 shape: {output.shape}")
                    
                    # 检查输出是否为空
                    if output.numel() == 0:
                        raise ValueError("模型生成了空输出。请检查输入形状或模型结构。")
                    
                    # 调整输出和目标维度
                    output = output.view(-1, self.feats)
                    target = target.view(-1, self.feats)
                    print(f"调整后的输出 shape: {output.shape}")
                    print(f"调整后的目标 shape: {target.shape}")
                    
                    # 计算异常分数
                    mse = torch.sub(output, target).pow(2)
                    scores.append(mse.cpu())
                    outputs.append(output.cpu())
                    print(f"批次异常分数 shape: {mse.shape}")
            
            if len(scores) == 0:
                raise ValueError("模型没有生成任何输出。请检查输入形状或提前停止条件。")
            
            if len(outputs) == 0:
                raise ValueError("模型没有生成任何输出。请检查输入形状或模型结构。")
            
            scores = torch.cat(scores, dim=0)
            scores = scores.numpy()
            scores = np.mean(scores, axis=1)
            
            if scores.shape[0] < len(data):
                padded_decision_scores_ = np.zeros(len(data))
                padded_decision_scores_[:self.window_size+self.pred_len-1] = scores[0]
                padded_decision_scores_[self.window_size+self.pred_len-1:] = scores
            else:
                padded_decision_scores_ = scores[:len(data)]
            
            print(f"\n预测完成:")
            print(f"最终输出维度: {padded_decision_scores_.shape}")
            print(f"输出统计:")
            print(f"- 最小值: {padded_decision_scores_.min():.6f}")
            print(f"- 最大值: {padded_decision_scores_.max():.6f}")
            print(f"- 均值: {padded_decision_scores_.mean():.6f}")
            print(f"- 标准差: {padded_decision_scores_.std():.6f}")
            
            self.__anomaly_score = padded_decision_scores_
            return padded_decision_scores_
            
        except Exception as e:
            error_msg = f"模型预测过程中出错: {str(e)}"
            print(error_msg)
            self.__anomaly_score = error_msg
            return error_msg
    
    def anomaly_score(self) -> np.ndarray:
        if isinstance(self.__anomaly_score, str):
            print("模型出错，未返回合法分数:", self.__anomaly_score)
            return np.zeros(1)  # 返回一个零数组
        return self.__anomaly_score 