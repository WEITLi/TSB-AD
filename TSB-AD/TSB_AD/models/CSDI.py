import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ForecastDataset

class CSDI(nn.Module):
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
        self.encoder = nn.Sequential(
            nn.Linear(feats, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, feats)
        )
        
        # 添加时间嵌入
        self.time_embed = nn.Embedding(win_size, 32)
        
        # 添加位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, win_size, 32))
        
        # self.model = self.to(self.device)
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.early_stopping = EarlyStoppingTorch(patience=3)
        
    def forward(self, x):
        # 时间嵌入
        time_idx = torch.arange(self.win_size, device=x.device)
        time_emb = self.time_embed(time_idx)
        
        # 位置编码
        pos_emb = self.pos_embed.expand(x.size(0), -1, -1)
        
        # 特征编码
        x = self.encoder(x)
        
        # 添加时间信息
        x = x + time_emb + pos_emb
        
        # 解码
        x = self.decoder(x)
        return x
        
    def fit(self, data):
        # 确保数据是浮点型
        data = data.astype(np.float32)
        
        # 限制数据长度
        max_length = 5000  # 减小最大长度
        if len(data) > max_length:
            data = data[:max_length]
        
        # 使用较小的 batch_size
        actual_batch_size = min(self.batch_size, len(data) // 2)
        if actual_batch_size < 1:
            actual_batch_size = 1
            
        train_loader = DataLoader(
            ForecastDataset(data, window_size=self.win_size, pred_len=1),
            batch_size=actual_batch_size,
            shuffle=True,
            num_workers=0  # 避免多进程
        )
        
        for epoch in range(1, self.epochs + 1):
            self.train()
            train_loss = 0
            for x, target in train_loader:
                x, target = x.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self(x)
                
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            self.scheduler.step()
            
            self.early_stopping(train_loss, self)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
    
    def decision_function(self, data):
        # 确保数据是浮点型
        data = data.astype(np.float32)
        
        # 限制数据长度
        max_length = 5000  # 减小最大长度
        if len(data) > max_length:
            data = data[:max_length]
        
        # 使用较小的 batch_size
        actual_batch_size = min(self.batch_size, len(data) // 2)
        if actual_batch_size < 1:
            actual_batch_size = 1
            
        test_loader = DataLoader(
            ForecastDataset(data, window_size=self.win_size, pred_len=1),
            batch_size=actual_batch_size,
            shuffle=False,
            num_workers=0  # 避免多进程
        )
        
        self.eval()
        scores = []
        with torch.no_grad():
            for x, target in test_loader:
                x, target = x.to(self.device), target.to(self.device)
                output = self(x)
                
                mse = torch.sub(output, target).pow(2)
                scores.append(mse.cpu())
        
        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()
        scores = np.mean(scores, axis=1)
        
        if scores.shape[0] < len(data):
            padded_decision_scores_ = np.zeros(len(data))
            padded_decision_scores_[:self.win_size] = scores[0]
            padded_decision_scores_[self.win_size:] = scores
        
        self.__anomaly_score = padded_decision_scores_
        return padded_decision_scores_
    
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score 