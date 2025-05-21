from typing import Dict
import torchinfo
import tqdm, math
import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import weight_norm # 导入 weight_norm
from torch.utils.data import DataLoader

from ..utils.utility import get_activation_by_name
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ForecastDataset

# TCN 的基本模块
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 第一层卷积 + ReLU + Dropout
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0) # 移除右侧填充以实现因果卷积
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层卷积 + ReLU + Dropout
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0) # 移除右侧填充
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 将上述层按顺序组合
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # 下采样层，如果输入输出通道数不同，需要通过 1x1 卷积调整
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # 初始化权重
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 残差连接
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# TCN 模型主体
class TCNModel(nn.Module):
    def __init__(self, n_features, num_channels, kernel_size=2, dropout=0.2, predict_time_steps=1, device='cpu'):
        super(TCNModel, self).__init__()
        self.n_features = n_features
        self.predict_time_steps = predict_time_steps
        self.device = device
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i # 膨胀系数按层数指数增长
            in_channels = n_features if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # 计算因果卷积所需的 padding
            # padding = (kernel_size - 1) * dilation_size
            # 注意：PyTorch 的 Conv1d padding 是在两侧进行的，我们需要在 chomp 层移除右侧 padding
            # 因此这里设置的 padding 应该是 (kernel_size - 1) * dilation_size
            causal_padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=causal_padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        # 输出层，将最后一个 TCN 块的输出映射到预测步长
        self.fc = nn.Linear(num_channels[-1], n_features * predict_time_steps)

    def forward(self, x):
        # 输入 x 的 shape: [batch_size, sequence_length, n_features]
        # TCN 需要 [batch_size, n_features, sequence_length]
        x = x.permute(0, 2, 1) 
        output = self.network(x)
        # 取序列的最后一个时间步的输出用于预测
        output = self.fc(output[:, :, -1]) 
        # 调整输出形状以匹配目标: [batch_size, predict_time_steps, n_features]
        output = output.view(x.size(0), self.predict_time_steps, self.n_features)
        # 再调整回 [predict_time_steps, batch_size, n_features] 以兼容原有 CNN 代码逻辑
        output = output.permute(1, 0, 2) 
        return output

# TCN 训练和评估封装类
class TCN():
    def __init__(self,
                 window_size=100,
                 pred_len=1,
                 batch_size=128,
                 epochs=50,
                 lr=0.001, # TCN 可能需要不同的学习率
                 feats=1,
                 num_channel=[25, 25, 25], # TCN 的通道数可以调整
                 kernel_size=3, # TCN 内核大小
                 dropout=0.2,   # TCN dropout 率
                 validation_size=0.2):
        super().__init__()
        self.__anomaly_score = None
        
        cuda = True # 假设使用 CUDA
        self.y_hats = None
        
        self.cuda = cuda
        self.device = get_gpu(self.cuda)
        
        self.window_size = window_size
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.feats = feats
        self.num_channel = num_channel
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.validation_size = validation_size
        
        # 实例化 TCNModel
        self.model = TCNModel(n_features=feats, 
                              num_channels=num_channel, 
                              kernel_size=kernel_size,
                              dropout=dropout,
                              predict_time_steps=self.pred_len, 
                              device=self.device).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # 可以根据需要调整学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9) 
        self.loss = nn.MSELoss()
        self.save_path = None # 可以设置模型保存路径
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=5) # TCN 可能需要更多 patience
        
        self.mu = None
        self.sigma = None
        self.eps = 1e-10
        
    def fit(self, data):
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            ForecastDataset(tsTrain, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=True)
        
        valid_loader = DataLoader(
            ForecastDataset(tsValid, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False)
        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                
                # TCN 模型输出已经是 [pred_len, batch_size, feats]
                output = self.model(x) 
                
                # 调整形状以计算损失
                # output: [pred_len, bs, feats] -> [bs, pred_len, feats] -> [bs, pred_len * feats]
                # 使用 x.size(0) 获取当前实际的批次大小
                current_batch_size = x.size(0)
                output_reshaped = output.permute(1, 0, 2).contiguous().view(current_batch_size, -1)
                # target: [bs, pred_len, feats] -> [bs, pred_len * feats]
                target_reshaped = target.view(current_batch_size, -1)

                # 确保重塑后的元素总数匹配
                if output_reshaped.numel() != target_reshaped.numel():
                    print(f"警告: 训练中形状不匹配! Output: {output_reshaped.shape}, Target: {target_reshaped.shape}")
                    # 可以选择跳过此批次或采取其他错误处理
                    continue 

                loss = self.loss(output_reshaped, target_reshaped)
                loss.backward()

                # 添加梯度裁剪，防止梯度爆炸 (对 TCN 可能有帮助)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) 

                self.optimizer.step()
                
                avg_loss += loss.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            
            self.model.eval()
            scores = []
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)

                    output = self.model(x) # [pred_len, bs, feats]

                    # 调整形状以计算损失和分数
                    # output_for_loss: [pred_len, bs, feats] -> [bs, pred_len, feats] -> [bs, pred_len * feats]
                    output_for_loss = output.permute(1, 0, 2).contiguous().view(x.size(0), -1)
                    # target_for_loss: [bs, pred_len, feats] -> [bs, pred_len * feats]
                    target_for_loss = target.view(x.size(0), -1)
                    
                    loss = self.loss(output_for_loss, target_for_loss)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    
                    # 计算逐点的 MSE 分数
                    # output: [pred_len, bs, feats] -> [bs, pred_len, feats]
                    # target: [bs, pred_len, feats]
                    mse = torch.sub(output.permute(1, 0, 2), target).pow(2) # shape: [bs, pred_len, feats]
                    # 对 pred_len 和 feats 维度求平均，得到每个样本的分数
                    mse = torch.mean(mse, dim=(1, 2)) # shape: [bs]
                    scores.append(mse.cpu())
                    
            
            valid_loss = avg_loss/max(len(valid_loader), 1)
            self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop or epoch == self.epochs: # 最后一个 epoch 也计算高斯分布
                # fitting Gaussian Distribution
                if len(scores) > 0:
                    scores_tensor = torch.cat(scores, dim=0) # shape: [num_valid_samples]
                    if scores_tensor.numel() > 0: # 确保张量不为空
                        self.mu = torch.mean(scores_tensor)
                        self.sigma = torch.var(scores_tensor)
                        # print(f"Mu: {self.mu.item()}, Sigma: {self.sigma.item()}") # 调试信息
                    else:
                        print("警告: 验证分数为空，无法拟合高斯分布。")
                        self.mu = torch.tensor(0.0) # 默认值
                        self.sigma = torch.tensor(1.0) # 默认值
                else:
                     print("警告: 验证分数列表为空，无法拟合高斯分布。")
                     self.mu = torch.tensor(0.0)
                     self.sigma = torch.tensor(1.0)

                if self.early_stopping.early_stop:
                    print("   Early stopping<<<")
                break

    def decision_function(self, data):
        test_loader = DataLoader(
            ForecastDataset(data, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # 如果模型未训练（mu 或 sigma 为 None），则先训练
        if self.mu is None or self.sigma is None:
            print("警告: 模型似乎未训练 (mu/sigma is None)，将尝试在测试集上训练模型。")
            # 注意：这通常不是好的做法，应该先调用 fit。这里仅作备用。
            self.fit(data) 
            # 再次检查 mu 和 sigma
            if self.mu is None or self.sigma is None:
                 print("错误：即使在尝试训练后，mu/sigma 仍然为 None。无法计算决策分数。")
                 return np.zeros(len(data)) # 返回零数组或引发错误

        self.model.eval()
        scores = []
        y_hats = [] # 存储预测值（如果需要）
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x) # [pred_len, bs, feats]

                # 调整形状以计算分数
                # output: [pred_len, bs, feats] -> [bs, pred_len, feats]
                output_permuted = output.permute(1, 0, 2)
                # target: [bs, pred_len, feats]
                
                # 计算逐点 MSE 分数
                mse = torch.sub(output_permuted, target).pow(2) # shape: [bs, pred_len, feats]
                # 对 pred_len 和 feats 维度求平均
                mse = torch.mean(mse, dim=(1, 2)) # shape: [bs]

                # y_hats.append(output_permuted.cpu()) # 如果需要存储预测
                scores.append(mse.cpu())
                loop.set_description(f'Testing: ')

        scores_tensor = torch.cat(scores, dim=0) # shape: [num_test_samples]
        
        # 使用训练/验证阶段得到的 mu 和 sigma 计算异常分数
        # Z-score or Mahalanobis distance like score (using variance)
        anomaly_scores = (scores_tensor - self.mu) / torch.sqrt(self.sigma + self.eps)
        # 或者使用原始的 MSE 分数，取决于如何定义异常分数
        # anomaly_scores = scores_tensor
        
        scores_np = anomaly_scores.numpy()
        
        # 填充分数以匹配原始数据长度
        # 注意：填充方式可能需要根据具体需求调整
        padded_decision_scores_ = np.zeros(len(data))
        # 假设异常分数对应于窗口的末尾
        score_start_index = self.window_size + self.pred_len - 1
        if len(scores_np) > 0:
             # 将第一个分数填充到前面
             padded_decision_scores_[:score_start_index] = scores_np[0] 
             # 填充计算得到的分数
             effective_length = min(len(scores_np), len(data) - score_start_index)
             padded_decision_scores_[score_start_index : score_start_index + effective_length] = scores_np[:effective_length]
             # 如果分数比需要填充的位置少，用最后一个分数填充剩余部分
             if effective_length < len(scores_np):
                  # This case should ideally not happen if dataset/loader is correct
                  pass 
             elif score_start_index + effective_length < len(data):
                  padded_decision_scores_[score_start_index + effective_length:] = scores_np[-1]

        else:
             print("警告：测试阶段未生成任何分数。")
             # 保持 padded_decision_scores_ 为零数组

        self.__anomaly_score = padded_decision_scores_
        return padded_decision_scores_

    def anomaly_score(self) -> np.ndarray:
        # 返回之前计算好的异常分数
        if self.__anomaly_score is None:
            print("请先调用 decision_function 来计算异常分数。")
            return np.array([]) # 或者 None
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        # 如果需要，实现 y_hat 的获取逻辑
        # if self.y_hats is None:
        #    print("请先调用 decision_function 来计算预测值。")
        #    return None
        # return torch.cat(self.y_hats, dim=0).numpy()
        print("get_y_hat 尚未在此 TCN 实现中完全支持。")
        return None
    
    def param_statistic(self, save_file):
        # 获取模型参数统计信息
        try:
             # 需要一个示例输入形状
             # 注意：如果 batch_size 或 window_size 在 __init__ 后改变，这里可能需要更新
             example_input_shape = (self.batch_size, self.window_size, self.feats)
             model_stats = torchinfo.summary(self.model, input_size=example_input_shape, verbose=0)
             with open(save_file, 'w') as f:
                 f.write(str(model_stats))
        except Exception as e:
             print(f"生成参数统计信息时出错: {e}")
             # 可以选择写入错误信息到文件
             with open(save_file, 'w') as f:
                  f.write(f"无法生成参数统计信息: {e}") 