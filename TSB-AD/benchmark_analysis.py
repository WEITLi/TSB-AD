import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入我们创建的模块
from statistical_analysis import prepare_benchmark_data, Friedman_Nemenyi
from visualization import plot_critical_diagram, plot_performance_boxplot

def analyze_benchmark_results(results_file, algorithms, metric='VUS-PR', filter_condition=None, 
                              alpha=0.05, title=None, save_dir=None):
    """
    分析基准测试结果，生成统计分析和可视化
    
    参数:
        results_file: 结果CSV文件路径
        algorithms: 要分析的算法列表
        metric: 要分析的性能指标 (默认 VUS-PR)
        filter_condition: 可选的过滤条件 (函数，接受DataFrame作为参数)
        alpha: 显著性水平
        title: 图形标题
        save_dir: 保存结果的目录
    """
    print(f"开始分析基准测试结果，指标: {metric}")
    
    # 加载数据
    try:
        df = pd.read_csv(results_file)
        print(f"已加载结果文件: {results_file}, 形状: {df.shape}")
    except Exception as e:
        print(f"加载结果文件时出错: {e}")
        return None, None
    
    # 应用过滤条件（如果提供）
    if filter_condition is not None:
        try:
            filtered_df = filter_condition(df)
            print(f"应用过滤条件后: {filtered_df.shape[0]} 行")
            df = filtered_df
        except Exception as e:
            print(f"应用过滤条件时出错: {e}")
            return None, None
    
    # 准备数据
    eval_df = prepare_benchmark_data(df, algorithms)
    print(f"准备好评估数据: {len(eval_df)} 条记录")
    
    # 绘制箱线图
    plt.figure(figsize=(8, 5))
    boxplot_title = f"{metric} 在不同算法上的分布" if title is None else f"{title}: {metric} 分布"
    
    # 确保只使用可用的算法列
    available_algorithms = [algo for algo in algorithms if algo in df.columns]
    if len(available_algorithms) < len(algorithms):
        print(f"警告: 只有 {len(available_algorithms)}/{len(algorithms)} 个算法在结果中可用")
    
    ax = plot_performance_boxplot(df, available_algorithms[:12], metric, title=boxplot_title)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{metric}_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 进行统计显著性分析并绘制临界差异图
    plt.figure(figsize=(10, 6))
    cd_title = f"{metric} 算法排名临界差异图" if title is None else f"{title}: {metric} 算法排名"
    
    # 调用可视化函数
    try:
        ranking, avg_ranks = plot_critical_diagram(eval_df, algorithms, title=cd_title, alpha=alpha)
    except Exception as e:
        print(f"绘制临界差异图时出错: {e}")
        plt.close()
        return None, None
    
    if save_dir and ranking is not None:
        plt.savefig(os.path.join(save_dir, f"{metric}_critical_diagram.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印排名结果
    if ranking:
        print("\n算法排名 (从高到低):")
        for i, algo in enumerate(ranking[:10]):
            rank_val = avg_ranks[algo] if avg_ranks is not None else "N/A"
            print(f"{i+1}. {algo}: {rank_val:.3f}")
    
    return ranking, avg_ranks

if __name__ == "__main__":
    # 示例用法
    
    # 定义要分析的算法列表 (根据你的结果文件调整)
    uni_algorithms = ['Sub-PCA', 'MOMENT (FT)', 'MOMENT (ZS)', 'POLY', 'CNN', 
                     'SR', 'Series2Graph', 'LSTMAD', 'IForest', 'USAD']
    
    # 多变量数据集算法
    multi_algorithms = ['IForest', 'LOF', 'PCA', 'HBOS', 'OCSVM', 'MCD', 'KNN', 
                      'KMeansAD', 'COPOD', 'CBLOF', 'EIF', 'RobustPCA', 'AutoEncoder']
    
    # 分析单变量数据集结果
    print("="*50)
    print("分析单变量数据集结果")
    print("="*50)
    
    uni_results_path = 'benchmark_eval_results/uni_mergedTable_VUS-PR.csv'
    # 分析所有数据
    uni_ranking, uni_ranks = analyze_benchmark_results(
        uni_results_path, 
        uni_algorithms,
        metric='VUS-PR',
        title="单变量数据集",
        save_dir='analysis_results'
    )
    
    # 仅分析有点异常的数据集
    # 使用函数而非直接引用df (修复bug)
    point_anomaly_ranking, _ = analyze_benchmark_results(
        uni_results_path, 
        uni_algorithms,
        metric='VUS-PR',
        filter_condition=lambda df: df[df['point_anomaly'] == 1] if 'point_anomaly' in df.columns else df,
        title="单变量点异常数据集",
        save_dir='analysis_results'
    )
    
    # 分析多变量数据集结果
    print("="*50)
    print("分析多变量数据集结果")
    print("="*50)
    
    multi_results_path = 'benchmark_eval_results/multi_mergedTable_VUS-PR.csv'
    multi_ranking, multi_ranks = analyze_benchmark_results(
        multi_results_path, 
        multi_algorithms,
        metric='VUS-PR',
        title="多变量数据集",
        save_dir='analysis_results'
    )
    
    print("\n分析完成！结果已保存到 'analysis_results' 目录") 