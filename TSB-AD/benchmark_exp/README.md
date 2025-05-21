# 时间序列异常检测基准测试

本目录包含用于评估时间序列异常检测算法的基准测试代码。

## 主要功能

### 1. 运行基准测试 (`run_benchmark.py`)

运行多个时间序列异常检测算法，评估它们在单变量或多变量数据集上的性能。

**新增功能：**
- 支持多次运行实验进行稳定性分析
- 收集和保存模型参数量和大小信息
- 在单变量和多变量数据集上独立运行
- 自动保存统计信息和运行结果

**使用示例：**

```bash
# 在单变量和多变量数据集上运行基准测试，每个2次实验
python run_benchmark.py --num_runs 2 --dataset_type Both

# 仅在单变量数据集上运行
python run_benchmark.py --num_runs 2 --dataset_type Univariate

# 仅在多变量数据集上运行
python run_benchmark.py --num_runs 2 --dataset_type Multivariate 
```

**参数说明：**

- `--uni_dataset`: 单变量数据集文件名
- `--multi_dataset`: 多变量数据集文件名
- `--uni_dataset_dir`: 单变量数据集目录
- `--multi_dataset_dir`: 多变量数据集目录
- `--save_dir`: 保存结果的目录
- `--log_file`: 日志文件路径
- `--num_runs`: 运行实验的次数（默认2次）
- `--dataset_type`: 数据集类型（Univariate/Multivariate/Both）

### 2. 结果分析 (`benchmark_analysis.py`)

分析基准测试结果，生成可视化图表和统计分析。

**主要功能：**
- 加载和分析多次运行的实验结果
- 统计分析各算法在不同指标上的性能
- 生成稳定性分析（标准差、变异系数）
- 分析模型参数量和大小
- 绘制临界差异图、箱线图等可视化
- 生成综合PDF报告

**使用示例：**

```bash
# 分析指定目录中的所有结果
python benchmark_analysis.py --results_dir eval/benchmark --save_dir analysis_results

# 仅分析单变量数据集结果
python benchmark_analysis.py --results_dir eval/benchmark --save_dir analysis_results --dataset_type Univariate
```

**参数说明：**

- `--results_dir`: 结果目录路径
- `--save_dir`: 保存分析结果的目录
- `--dataset_type`: 要分析的数据集类型（Univariate/Multivariate/Both）
- `--alpha`: 统计显著性水平（默认0.05）

## 评估指标

基准测试评估使用以下指标：
- `AUC-PR`: 精准率-召回率曲线下面积
- `AUC-ROC`: ROC曲线下面积
- `VUS-PR`: Volume Under Surface for PR
- `VUS-ROC`: Volume Under Surface for ROC
- `Standard-F1`: 标准F1分数
- `PA-F1`: 点调整F1分数
- `Event-based-F1`: 事件级F1分数
- `R-based-F1`: R-based F1分数
- `Affiliation-F`: Affiliation F分数

## 算法列表

支持的算法按类型划分:

### 重构型
- AutoEncoder
- PCA
- USAD
- OmniAnomaly

### 预测型
- LSTMAD
- DLinear

### 插补型
- Donut

### 聚类型
- KMeansAD
- LOF
- OCSVM

### 混合型
- AnomalyTransformer

## 分析输出

运行`benchmark_analysis.py`后，将生成以下输出：
- 算法性能对比图表（箱线图）
- 临界差异图（算法排名）
- 稳定性分析（热力图、变异系数）
- 模型大小和参数量分析
- 综合PDF报告

## 注意事项

- 多次运行实验时，每次使用不同的随机种子以确保结果可靠性
- 分析结果时，默认会加载所有可用结果文件
- 可以通过 `--dataset_type` 参数选择仅分析单变量或多变量结果

### Scripts for running experiments/develop new methods in TSB-AD

* Hper-parameter Tuning: HP_Tuning_U/M.py

* Benchmark Evaluation: Run_Detector_U/M.py

* `benchmark_eval_results/`: Evaluation results of anomaly detectors across different time series in TSB-AD
    * All time series are normalized by z-score by default

* Develop your own algorithm: Run_Custom_Detector.py
    * Step 1: Implement `Custom_AD` class
    * Step 2: Implement model wrapper function `run_Custom_AD_Unsupervised` or `run_Custom_AD_Semisupervised`
    * Step 3: Specify `Custom_AD_HP` hyperparameter dict
    * Step 4: Run the custom algorithm either `run_Custom_AD_Unsupervised` or `run_Custom_AD_Semisupervised`
    * Step 5: Apply threshold to the anomaly score (if any)

🪧 How to commit your own algorithm to TSB-AD: you can send us the Run_Custom_Detector.py (replace Custom_Detector with the model name) to us via (i) [email](liu.11085@osu.edu) or (ii) open a pull request and add the file to `benchmark_exp` folder in `TSB-AD-algo` branch. We will test and evaluate the algorithm and include it in our [leaderboard](https://thedatumorg.github.io/TSB-AD/).

* Run_My_Detector.py: Run your own detector