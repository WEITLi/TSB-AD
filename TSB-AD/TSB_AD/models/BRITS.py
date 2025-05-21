import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ForecastDataset

class BRITS(nn.Module):
    def __init__(self, win_size, feats, lr=1e-3, batch_size=128, epochs=50):
        super().__init__()
        self.__anomaly_score = None
        
        cuda = True
        self.device = get_gpu(cuda)
        
        self.win_size = win_size
        self.feats = feats
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        
        # 定义模型结构
        self.encoder = nn.LSTM(
            input_size=feats,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.decoder = nn.LSTM(
            input_size=64,
            hidden_size=feats,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 添加投影层
        self.input_proj = nn.Linear(feats, feats)
        self.output_proj = nn.Linear(feats, feats)
        
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.early_stopping = EarlyStoppingTorch(patience=3)
        
    def forward(self, x):
        # 输入投影
        x = self.input_proj(x)
        
        # 编码
        encoded, _ = self.encoder(x)
        
        # 解码
        decoded, _ = self.decoder(encoded)
        
        # 输出投影
        decoded = self.output_proj(decoded)
        return decoded
        
    def fit(self, data):
        # 确保数据是浮点型
        data = data.astype(np.float32)
        
        # 限制数据长度
        if len(data) > 2000:
            data = data[:2000]
        
        # 计算实际batch size
        actual_batch_size = min(self.batch_size, len(data) // 4)
        if actual_batch_size < 1:
            actual_batch_size = 1
        
        train_loader = DataLoader(
            ForecastDataset(data, window_size=self.win_size, pred_len=1),
            batch_size=actual_batch_size,
            shuffle=True,
            num_workers=0
        )
        
        for epoch in range(1, self.epochs + 1):
            self.train()
            train_loss = 0
            for x, target in train_loader:
                x, target = x.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self(x)
                
                # 确保输出和目标维度匹配
                output = output[:, -1:, :]  # 只取最后一个时间步
                target = target.unsqueeze(1)  # 添加时间维度
                loss = self.loss(output, target)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.scheduler.step()
            
            self.early_stopping(train_loss, self)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
    
    def decision_function(self, data):
        # 确保数据是浮点型
        data = data.astype(np.float32)
        
        # 限制数据长度
        if len(data) > 2000:
            data = data[:2000]
        
        # 计算实际batch size
        actual_batch_size = min(self.batch_size, len(data) // 4)
        if actual_batch_size < 1:
            actual_batch_size = 1
        
        test_loader = DataLoader(
            ForecastDataset(data, window_size=self.win_size, pred_len=1),
            batch_size=actual_batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.eval()
        scores = []
        with torch.no_grad():
            for x, target in test_loader:
                x, target = x.to(self.device), target.to(self.device)
                output = self(x)
                
                # 确保输出和目标维度匹配
                output = output[:, -1:, :]  # 只取最后一个时间步
                target = target.unsqueeze(1)  # 添加时间维度
                mse = torch.sub(output, target).pow(2)
                scores.append(mse.cpu())
        
        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()
        
        # 处理维度
        if len(scores.shape) == 3:
            scores = np.mean(scores, axis=(1, 2))
        elif len(scores.shape) == 2:
            scores = np.mean(scores, axis=1)
        
        # 确保输出是一维数组
        scores = scores.ravel()
        
        # 处理输出长度
        if scores.shape[0] < len(data):
            padded_decision_scores_ = np.zeros(len(data))
            padded_decision_scores_[:self.win_size] = scores[0]
            padded_decision_scores_[self.win_size:] = scores
        else:
            padded_decision_scores_ = scores[:len(data)]  # 确保长度匹配
        
        self.__anomaly_score = padded_decision_scores_
        return padded_decision_scores_
    
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score 