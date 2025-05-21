from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

# determine sliding window (period) based on ACF
def find_length_rank(data, rank=1):
    """
    基于自相关函数(ACF)确定滑动窗口大小，支持多个周期
    
    参数:
    data: 输入的时间序列数据
    rank: 要选择的周期排名（1表示主要周期，2表示次要周期，3表示第三周期）
    
    返回:
    滑动窗口大小（周期长度）
    """
    # 确保数据是一维的
    data = data.squeeze()
    if len(data.shape)>1: return 0
    if rank==0: return 1
    
    # 限制数据长度，避免计算量过大
    data = data[:min(20000, len(data))]
    
    # 计算自相关函数
    base = 3  # 基础偏移量
    auto_corr = acf(data, nlags=400, fft=True)[base:]  # 计算自相关，使用FFT加速
    
    # 找到局部最大值点
    local_max = argrelextrema(auto_corr, np.greater)[0]
    
    try:
        # 按自相关值大小排序局部最大值点
        sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]
        
        # 根据rank选择不同的周期
        if rank == 1:  # 主要周期
            max_local_max = sorted_local_max[0]
        elif rank == 2:  # 次要周期
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    max_local_max = i 
                    break
        elif rank == 3:  # 第三周期
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    id_tmp = i
                    break
            for i in sorted_local_max[id_tmp:]:
                if i > sorted_local_max[id_tmp]: 
                    max_local_max = i           
                    break
        
        # 检查周期是否在合理范围内
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125  # 默认值
        return local_max[max_local_max]+base
    except:
        return 125  # 出错时返回默认值
    

# determine sliding window (period) based on ACF, Original version
def find_length(data):
    """
    基于自相关函数(ACF)确定滑动窗口大小的原始版本，只返回主要周期
    
    参数:
    data: 输入的时间序列数据
    
    返回:
    滑动窗口大小（周期长度）
    """
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        # 只选择自相关值最大的点
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125
