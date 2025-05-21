import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ForecastDataset

class SAITS(nn.Module):
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
        
        print(f"初始化 SAITS 模型:")
        print(f"- 窗口大小: {win_size}")
        print(f"- 特征数: {feats}")
        print(f"- 批次大小: {batch_size}")
        print(f"- 训练轮数: {epochs}")
        
        # 确保嵌入维度能被注意力头数量整除
        self.num_heads = 4
        self.embed_dim = feats
        if self.embed_dim % self.num_heads != 0:
            self.embed_dim = (self.embed_dim // self.num_heads + 1) * self.num_heads
            print(f"- 调整后的嵌入维度: {self.embed_dim}")
        
        # 定义模型结构
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # 添加投影层
        self.input_proj = nn.Linear(feats, self.embed_dim)  # 从原始特征维度投影到嵌入维度
        self.output_proj = nn.Linear(self.embed_dim, feats)  # 从嵌入维度投影回原始特征维度
        
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.early_stopping = EarlyStoppingTorch(patience=3)
        
    def forward(self, x):
        # 输入维度检查
        print(f"Forward 输入维度: {x.shape}")
        
        # 输入投影
        x = self.input_proj(x)  # [batch_size, seq_len, feats] -> [batch_size, seq_len, embed_dim]
        print(f"投影后维度: {x.shape}")
        
        # 创建掩码
        mask = self._generate_square_subsequent_mask(x.size(1)).to(self.device)
        print(f"掩码维度: {mask.shape}")
        
        # 编码
        memory = self.encoder(x, mask=mask)  # [batch_size, seq_len, embed_dim]
        print(f"编码器输出维度: {memory.shape}")
        
        # 解码
        output = self.decoder(x, memory, tgt_mask=mask)  # [batch_size, seq_len, embed_dim]
        print(f"解码器输出维度: {output.shape}")
        
        # 输出投影
        output = self.output_proj(output)  # [batch_size, seq_len, feats]
        print(f"最终输出维度: {output.shape}")
        
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def fit(self, data):
        print(f"\n开始训练:")
        print(f"输入数据维度: {data.shape}")
        print(f"输入数据类型: {data.dtype}")
        print(f"输入数据范围: [{data.min():.4f}, {data.max():.4f}]")
        
        # 确保数据是浮点型
        data = data.astype(np.float32)
        
        # 限制数据长度
        if len(data) > 2000:
            print(f"数据长度超过2000，截取前2000个样本")
            data = data[:2000]
            print(f"截取后数据维度: {data.shape}")
        
        # 计算实际batch size
        actual_batch_size = min(self.batch_size, len(data) // 4)
        if actual_batch_size < 1:
            actual_batch_size = 1
        print(f"实际使用的batch size: {actual_batch_size}")
        
        # 创建数据集
        dataset = ForecastDataset(data, window_size=self.win_size, pred_len=1)
        print(f"数据集大小: {len(dataset)}")
        print(f"数据集第一个样本:")
        x, target = dataset[0]
        print(f"- 输入维度: {x.shape}")
        print(f"- 目标维度: {target.shape}")
        
        train_loader = DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            num_workers=0
        )
        print(f"训练数据加载器大小: {len(train_loader)}")
        
        for epoch in range(1, self.epochs + 1):
            self.train()
            train_loss = 0
            for batch_idx, (x, target) in enumerate(train_loader):
                print(f"\n批次 {batch_idx + 1}:")
                print(f"输入维度: {x.shape}")
                print(f"目标维度: {target.shape}")
                
                x, target = x.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self(x)
                
                # 确保输出和目标维度匹配
                output = output[:, -1:, :]  # 只取最后一个时间步 [batch_size, 1, feats]
                print(f"调整后的输出维度: {output.shape}")
                print(f"目标维度: {target.shape}")
                
                # 计算损失
                loss = self.loss(output, target)
                print(f"当前批次损失: {loss.item():.4f}")
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = train_loss / len(train_loader)
            print(f"\nEpoch {epoch}/{self.epochs} - 平均损失: {avg_loss:.4f}")
            
            self.scheduler.step()
            
            self.early_stopping(avg_loss, self)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
    
    def decision_function(self, data):
        print(f"\n开始预测:")
        print(f"输入数据维度: {data.shape}")
        print(f"输入数据类型: {data.dtype}")
        print(f"输入数据范围: [{data.min():.4f}, {data.max():.4f}]")
        
        # 确保数据是浮点型
        data = data.astype(np.float32)
        
        # 限制数据长度
        if len(data) > 2000:
            print(f"数据长度超过2000，截取前2000个样本")
            data = data[:2000]
            print(f"截取后数据维度: {data.shape}")
        
        # 计算实际batch size
        actual_batch_size = min(self.batch_size, len(data) // 4)
        if actual_batch_size < 1:
            actual_batch_size = 1
        print(f"实际使用的batch size: {actual_batch_size}")
        
        # 创建数据集
        dataset = ForecastDataset(data, window_size=self.win_size, pred_len=1)
        print(f"数据集大小: {len(dataset)}")
        print(f"数据集第一个样本:")
        x, target = dataset[0]
        print(f"- 输入维度: {x.shape}")
        print(f"- 目标维度: {target.shape}")
        
        test_loader = DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=False,
            num_workers=0
        )
        print(f"测试数据加载器大小: {len(test_loader)}")
        
        self.eval()
        scores = []
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_loader):
                print(f"\n批次 {batch_idx + 1}:")
                print(f"输入维度: {x.shape}")
                print(f"目标维度: {target.shape}")
                
                x, target = x.to(self.device), target.to(self.device)
                output = self(x)
                
                # 确保输出和目标维度匹配
                output = output[:, -1:, :]  # 只取最后一个时间步 [batch_size, 1, feats]
                print(f"调整后的输出维度: {output.shape}")
                print(f"目标维度: {target.shape}")
                
                # 计算 MSE
                mse = torch.sub(output, target).pow(2)
                print(f"MSE 维度: {mse.shape}")
                scores.append(mse.cpu())
        
        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()
        print(f"合并后的分数维度: {scores.shape}")
        
        # 处理维度
        if len(scores.shape) == 3:
            scores = np.mean(scores, axis=(1, 2))
        elif len(scores.shape) == 2:
            scores = np.mean(scores, axis=1)
        print(f"降维后的分数维度: {scores.shape}")
        
        # 确保输出是一维数组
        scores = scores.ravel()
        print(f"展平后的分数维度: {scores.shape}")
        
        # 处理输出长度
        if scores.shape[0] < len(data):
            print(f"分数长度 ({scores.shape[0]}) 小于数据长度 ({len(data)})，进行填充")
            padded_decision_scores_ = np.zeros(len(data))
            padded_decision_scores_[:self.win_size] = scores[0]
            padded_decision_scores_[self.win_size:] = scores
        else:
            print(f"分数长度 ({scores.shape[0]}) 大于等于数据长度 ({len(data)})，进行截断")
            padded_decision_scores_ = scores[:len(data)]  # 确保长度匹配
        
        print(f"最终分数维度: {padded_decision_scores_.shape}")
        
        self.__anomaly_score = padded_decision_scores_
        return padded_decision_scores_
    
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score 

    def quick_test(self, data, test_size=100, n_epochs=2, n_batches=2):
        """快速测试函数，用于验证模型的基本功能
        
        Args:
            data: 输入数据
            test_size: 测试数据大小
            n_epochs: 测试用的训练轮数
            n_batches: 每个epoch测试的批次数
        """
        print("\n=== 开始快速测试 ===")
        
        # 1. 数据准备
        print("\n1. 数据准备:")
        data = data.astype(np.float32)
        if len(data) > test_size:
            data = data[:test_size]
        print(f"- 数据维度: {data.shape}")
        print(f"- 数据类型: {data.dtype}")
        print(f"- 数据范围: [{data.min():.4f}, {data.max():.4f}]")
        
        # 2. 数据集创建
        print("\n2. 数据集创建:")
        dataset = ForecastDataset(data, window_size=self.win_size, pred_len=1)
        print(f"- 数据集大小: {len(dataset)}")
        x, target = dataset[0]
        print(f"- 第一个样本:")
        print(f"  * 输入维度: {x.shape}")
        print(f"  * 目标维度: {target.shape}")
        
        # 3. 数据加载器
        print("\n3. 数据加载器:")
        loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
        print(f"- 批次数量: {len(loader)}")
        
        # 4. 模型测试
        print("\n4. 模型测试:")
        self.train()
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            for i, (x, target) in enumerate(loader):
                if i >= n_batches:
                    break
                    
                print(f"\n批次 {i + 1}:")
                print(f"- 输入维度: {x.shape}")
                print(f"- 目标维度: {target.shape}")
                
                x, target = x.to(self.device), target.to(self.device)
                
                # 前向传播
                output = self(x)
                print(f"- 模型输出维度: {output.shape}")
                
                # 调整输出维度
                output = output[:, -1:, :]
                print(f"- 调整后输出维度: {output.shape}")
                
                # 计算损失
                loss = self.loss(output, target)
                print(f"- 损失值: {loss.item():.4f}")
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 检查梯度
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float('inf'))
                print(f"- 梯度范数: {grad_norm:.4f}")
        
        # 5. 预测测试
        print("\n5. 预测测试:")
        self.eval()
        with torch.no_grad():
            x, target = next(iter(loader))
            x, target = x.to(self.device), target.to(self.device)
            
            output = self(x)
            output = output[:, -1:, :]
            
            print(f"- 输入维度: {x.shape}")
            print(f"- 目标维度: {target.shape}")
            print(f"- 输出维度: {output.shape}")
            print(f"- 预测损失: {self.loss(output, target).item():.4f}")
        
        print("\n=== 快速测试完成 ===")
        return True

def run_SAITS(data_train, data_test, win_size=100, lr=1e-3, batch_size=128, epochs=50):
    """运行SAITS模型的包装函数"""
    print(f"初始化SAITS模型...")
    print(f"训练数据维度: {data_train.shape}")
    print(f"测试数据维度: {data_test.shape}")
    
    # 确保数据维度正确
    if len(data_train.shape) == 1:
        data_train = data_train.reshape(-1, 1)
    if len(data_test.shape) == 1:
        data_test = data_test.reshape(-1, 1)
    
    # 调整batch_size以适应数据大小
    actual_batch_size = min(batch_size, len(data_train) // 4)
    if actual_batch_size < 1:
        actual_batch_size = 1
    print(f"实际使用的batch size: {actual_batch_size}")
    
    # 初始化模型
    model = SAITS(
        win_size=win_size,
        feats=data_test.shape[1],  # 使用实际的特征数
        lr=lr,
        batch_size=actual_batch_size,
        epochs=epochs
    )
    
    # 快速测试
    print(f"执行快速测试...")
    test_result = model.quick_test(data_train, test_size=200, n_epochs=2, n_batches=2)
    
    if not test_result:
        print("快速测试失败，终止训练")
        return np.zeros(len(data_test))
    
    # 正常训练
    print(f"开始正常训练...")
    model.fit(data_train)
    
    # 预测
    print(f"开始预测...")
    score = model.decision_function(data_test)
    
    # 确保输出维度与输入数据长度一致
    if len(score) != len(data_test):
        print(f"警告: 预测分数长度 ({len(score)}) 与测试数据长度 ({len(data_test)}) 不匹配")
        print("调整预测分数长度...")
        if len(score) < len(data_test):
            # 如果分数长度小于数据长度，进行填充
            padded_score = np.zeros(len(data_test))
            padded_score[:len(score)] = score
            score = padded_score
        else:
            # 如果分数长度大于数据长度，进行截断
            score = score[:len(data_test)]
    
    print(f"最终输出维度: {score.shape}")
    return score.ravel() 