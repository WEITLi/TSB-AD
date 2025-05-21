import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class DBSCAN_AD:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, 
                           algorithm=algorithm, leaf_size=leaf_size)
        self.scaler = StandardScaler()
        self.__anomaly_score = None
        
    def fit(self, data):
        try:
            print(f"DBSCAN fit - Input data shape: {data.shape}")
            
            # 检查数据是否包含 NaN 或 inf
            if np.isnan(data).any() or np.isinf(data).any():
                print("Warning: Input data contains NaN or inf values. Replacing with 0.")
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 标准化数据
            data_scaled = self.scaler.fit_transform(data)
            print(f"DBSCAN fit - Scaled data shape: {data_scaled.shape}")
            
            # 训练 DBSCAN 模型
            self.model.fit(data_scaled)
            print(f"DBSCAN fit - Number of clusters: {len(set(self.model.labels_)) - (1 if -1 in self.model.labels_ else 0)}")
            print(f"DBSCAN fit - Number of noise points: {list(self.model.labels_).count(-1)}")
            
            # 计算异常分数
            distances = []
            if hasattr(self.model, 'components_') and len(self.model.components_) > 0:
                for point in data_scaled:
                    min_dist = float('inf')
                    for core_point in self.model.components_:
                        dist = np.linalg.norm(point - core_point)
                        min_dist = min(min_dist, dist)
                    distances.append(min_dist)
            else:
                print("Warning: No core points found in DBSCAN. Using distance to nearest neighbor.")
                # 如果没有核心点，使用到最近邻的距离
                for i, point in enumerate(data_scaled):
                    min_dist = float('inf')
                    for j, other_point in enumerate(data_scaled):
                        if i != j:
                            dist = np.linalg.norm(point - other_point)
                            min_dist = min(min_dist, dist)
                    distances.append(min_dist)
            
            # 将距离转换为异常分数
            self.__anomaly_score = np.array(distances)
            print(f"DBSCAN fit - Anomaly scores shape: {self.__anomaly_score.shape}")
            
            return self
            
        except Exception as e:
            print(f"Error in DBSCAN fit: {str(e)}")
            raise
        
    def decision_scores_(self):
        if self.__anomaly_score is None:
            raise ValueError("Model has not been fitted yet.")
        return self.__anomaly_score
        
    def decision_function(self, data):
        try:
            print(f"DBSCAN decision_function - Input data shape: {data.shape}")
            
            # 检查数据是否包含 NaN 或 inf
            if np.isnan(data).any() or np.isinf(data).any():
                print("Warning: Input data contains NaN or inf values. Replacing with 0.")
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 标准化数据
            data_scaled = self.scaler.transform(data)
            print(f"DBSCAN decision_function - Scaled data shape: {data_scaled.shape}")
            
            # 获取每个点到最近核心点的距离
            distances = []
            if hasattr(self.model, 'components_') and len(self.model.components_) > 0:
                for point in data_scaled:
                    min_dist = float('inf')
                    for core_point in self.model.components_:
                        dist = np.linalg.norm(point - core_point)
                        min_dist = min(min_dist, dist)
                    distances.append(min_dist)
            else:
                print("Warning: No core points found in DBSCAN. Using distance to nearest neighbor.")
                # 如果没有核心点，使用到最近邻的距离
                for i, point in enumerate(data_scaled):
                    min_dist = float('inf')
                    for j, other_point in enumerate(data_scaled):
                        if i != j:
                            dist = np.linalg.norm(point - other_point)
                            min_dist = min(min_dist, dist)
                    distances.append(min_dist)
            
            # 将距离转换为异常分数
            self.__anomaly_score = np.array(distances)
            print(f"DBSCAN decision_function - Anomaly scores shape: {self.__anomaly_score.shape}")
            
            return self.__anomaly_score
            
        except Exception as e:
            print(f"Error in DBSCAN decision_function: {str(e)}")
            raise 