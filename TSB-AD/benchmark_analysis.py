import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from datetime import datetime
import json

# 导入我们创建的模块
from statistical_analysis import prepare_benchmark_data, Friedman_Nemenyi
from visualization import plot_critical_diagram, plot_performance_boxplot

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文字符
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def load_results(results_dir, dataset_type='Both'):
    """
    加载所有结果文件
    
    参数:
        results_dir: 结果目录
        dataset_type: 'Univariate', 'Multivariate', 或 'Both'
    
    返回:
        uni_df: 单变量数据
        multi_df: 多变量数据
        stats_uni_df: 单变量统计
        stats_multi_df: 多变量统计
    """
    uni_df, multi_df = None, None
    stats_uni_df, stats_multi_df = None, None
    
    if dataset_type in ['Univariate', 'Both']:
        # 尝试加载单变量统计结果
        uni_stats_path = os.path.join(results_dir, 'benchmark_stats_Univariate.csv')
        if os.path.exists(uni_stats_path):
            stats_uni_df = pd.read_csv(uni_stats_path)
            print(f"已加载单变量统计数据: {uni_stats_path}")
        
        # 尝试加载单变量组合结果
        uni_combined_path = os.path.join(results_dir, 'benchmark_results_Univariate_combined.csv')
        if os.path.exists(uni_combined_path):
            uni_df = pd.read_csv(uni_combined_path)
            print(f"已加载单变量合并数据: {uni_combined_path}")
    
    if dataset_type in ['Multivariate', 'Both']:
        # 尝试加载多变量统计结果
        multi_stats_path = os.path.join(results_dir, 'benchmark_stats_Multivariate.csv')
        if os.path.exists(multi_stats_path):
            stats_multi_df = pd.read_csv(multi_stats_path)
            print(f"已加载多变量统计数据: {multi_stats_path}")
            
        # 尝试加载多变量组合结果
        multi_combined_path = os.path.join(results_dir, 'benchmark_results_Multivariate_combined.csv')
        if os.path.exists(multi_combined_path):
            multi_df = pd.read_csv(multi_combined_path)
            print(f"已加载多变量合并数据: {multi_combined_path}")
    
    # 如果没有找到合并数据，尝试查找单个运行的结果
    if uni_df is None and dataset_type in ['Univariate', 'Both']:
        for run_id in range(1, 10):  # 最多查找10次运行
            path = os.path.join(results_dir, f'benchmark_results_Univariate_run{run_id}.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                if uni_df is None:
                    uni_df = df
                else:
                    uni_df = pd.concat([uni_df, df], ignore_index=True)
                print(f"已加载单变量运行{run_id}数据: {path}")
    
    if multi_df is None and dataset_type in ['Multivariate', 'Both']:
        for run_id in range(1, 10):  # 最多查找10次运行
            path = os.path.join(results_dir, f'benchmark_results_Multivariate_run{run_id}.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                if multi_df is None:
                    multi_df = df
                else:
                    multi_df = pd.concat([multi_df, df], ignore_index=True)
                print(f"已加载多变量运行{run_id}数据: {path}")
                
    return uni_df, multi_df, stats_uni_df, stats_multi_df

def load_model_details(results_dir, dataset_type='Both'):
    """加载模型详细信息JSON文件"""
    model_details = {}
    
    if dataset_type in ['Univariate', 'Both']:
        for run_id in range(1, 10):  # 最多查找10次运行
            path = os.path.join(results_dir, f'model_details_Univariate_run{run_id}.json')
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        # 为每个模型在不同运行中收集信息
                        for algo, details in data.items():
                            if algo not in model_details:
                                model_details[algo] = []
                            details['run_id'] = run_id
                            details['dataset_type'] = 'Univariate'
                            model_details[algo].append(details)
                    print(f"已加载单变量运行{run_id}模型详情: {path}")
                except Exception as e:
                    print(f"加载 {path} 时出错: {e}")
    
    if dataset_type in ['Multivariate', 'Both']:
        for run_id in range(1, 10):  # 最多查找10次运行
            path = os.path.join(results_dir, f'model_details_Multivariate_run{run_id}.json')
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        # 为每个模型在不同运行中收集信息
                        for algo, details in data.items():
                            if algo not in model_details:
                                model_details[algo] = []
                            details['run_id'] = run_id
                            details['dataset_type'] = 'Multivariate'
                            model_details[algo].append(details)
                    print(f"已加载多变量运行{run_id}模型详情: {path}")
                except Exception as e:
                    print(f"加载 {path} 时出错: {e}")
                    
    return model_details

def analyze_runtime_and_model_size(stats_df, save_dir=None, prefix=''):
    """分析运行时间和模型大小"""
    if stats_df is None or len(stats_df) == 0:
        print("没有可用的统计数据进行运行时间和模型大小分析")
        return
    
    # 提取可用的算法
    algos = list(stats_df['Algorithm'].unique())
    
    # 检查是否有模型大小信息
    has_model_size = 'ModelSizeMB' in stats_df.columns and stats_df['ModelSizeMB'].notna().any()
    has_total_params = 'TotalParams' in stats_df.columns and stats_df['TotalParams'].notna().any()
    
    # 创建数据框以绘图
    plot_data = []
    for _, row in stats_df.iterrows():
        algo = row['Algorithm']
        time_mean = row.get('Time_Mean', None)
        time_std = row.get('Time_Std', None)
        model_size = row.get('ModelSizeMB', None)
        total_params = row.get('TotalParams', None)
        
        if pd.notna(time_mean):
            plot_data.append({
                'Algorithm': algo,
                'Metric': '运行时间 (秒)',
                'Value': time_mean,
                'Std': time_std if pd.notna(time_std) else 0
            })
        
        if has_model_size and pd.notna(model_size) and model_size != 'NA':
            try:
                plot_data.append({
                    'Algorithm': algo,
                    'Metric': '模型大小 (MB)',
                    'Value': float(model_size),
                    'Std': 0  # 模型大小通常没有标准差
                })
            except:
                pass
                
        if has_total_params and pd.notna(total_params) and total_params != 'NA':
            try:
                plot_data.append({
                    'Algorithm': algo,
                    'Metric': '模型参数量',
                    'Value': float(total_params),
                    'Std': 0  # 参数量通常没有标准差
                })
            except:
                pass
    
    if not plot_data:
        print("没有可用的运行时间或模型大小数据进行绘图")
        return
        
    plot_df = pd.DataFrame(plot_data)
    
    # 为每种指标创建单独的图表
    for metric in plot_df['Metric'].unique():
        metric_df = plot_df[plot_df['Metric'] == metric]
        if len(metric_df) == 0:
            continue
            
        plt.figure(figsize=(12, 6))
        
        # 绘制条形图
        bars = plt.bar(metric_df['Algorithm'], metric_df['Value'], yerr=metric_df['Std'], 
                      capsize=5, color='skyblue', edgecolor='black')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(metric_df['Value']),
                    f'{height:.2f}', ha='center', va='bottom', rotation=0)
        
        plt.title(f"{metric} 分析")
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric)
        plt.tight_layout()
        
        # 保存图表
        if save_dir:
            save_path = os.path.join(save_dir, f"{prefix}_{metric.replace(' ', '_').replace('(', '').replace(')', '')}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"保存图表到: {save_path}")
    
        plt.show()

def analyze_metrics_stability(df, metrics=None, save_dir=None, prefix=''):
    """分析指标在多次运行中的稳定性"""
    if df is None or len(df) == 0:
        print("没有可用的数据进行稳定性分析")
        return
    
    if metrics is None:
        # 尝试找出所有指标列
        all_cols = df.columns
        metrics = [col for col in all_cols if col in DEFAULT_METRIC_NAMES]
    
    # 提取唯一的算法和运行次数
    algos = sorted(df['Algorithm'].unique())
    runs = sorted(df['Run'].unique())
    
    if len(runs) <= 1:
        print("只有一次运行，无法进行稳定性分析")
        return
    
    # 每个指标一个图
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        # 创建一个热力图数据
        heatmap_data = np.zeros((len(algos), len(runs)))
        has_valid_data = False
        
        for i, algo in enumerate(algos):
            for j, run in enumerate(runs):
                # 获取特定算法和运行的指标值
                value = df[(df['Algorithm'] == algo) & (df['Run'] == run)][metric]
                
                if len(value) > 0 and pd.notna(value.iloc[0]) and value.iloc[0] != 'RunError' and value.iloc[0] != 'EvalError':
                    try:
                        heatmap_data[i, j] = float(value.iloc[0])
                        has_valid_data = True
                    except:
                        heatmap_data[i, j] = np.nan
                else:
                    heatmap_data[i, j] = np.nan
        
        if not has_valid_data:
            print(f"指标 {metric} 没有足够的有效数据进行稳定性分析")
            continue
        
        # 计算每个算法的变异系数 (CV = std/mean)
        cv_values = []
        for i, algo in enumerate(algos):
            row_data = heatmap_data[i, :]
            if np.sum(~np.isnan(row_data)) > 1:  # 至少需要两个有效值
                cv = np.nanstd(row_data) / np.nanmean(row_data) if np.nanmean(row_data) != 0 else np.nan
                cv_values.append((algo, cv))
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        mask = np.isnan(heatmap_data)
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", 
                   xticklabels=[f"Run {r}" for r in runs], 
                   yticklabels=algos,
                   mask=mask)
        plt.title(f"{metric} 在多次运行中的稳定性")
        plt.tight_layout()
        
        # 保存热力图
        if save_dir:
            heatmap_path = os.path.join(save_dir, f"{prefix}_stability_{metric}_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            print(f"保存稳定性热力图到: {heatmap_path}")
            
        plt.show()
        
        # 绘制变异系数条形图
        if cv_values:
            cv_values.sort(key=lambda x: x[1])  # 按变异系数排序
            algos_sorted = [x[0] for x in cv_values]
            cv_sorted = [x[1] for x in cv_values]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(algos_sorted, cv_sorted, color='lightcoral', edgecolor='black')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.4f}', ha='center', va='bottom', rotation=0)
            
            plt.title(f"{metric} 算法稳定性 (变异系数 - 越低越稳定)")
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('变异系数 (CV = std/mean)')
            plt.tight_layout()
            
            # 保存变异系数图
            if save_dir:
                cv_path = os.path.join(save_dir, f"{prefix}_stability_{metric}_cv.png")
                plt.savefig(cv_path, dpi=300, bbox_inches='tight')
                print(f"保存变异系数图到: {cv_path}")
                
            plt.show()

def analyze_benchmark_results(df, algorithms, metric='VUS-PR', filter_condition=None, 
                             alpha=0.05, title=None, save_dir=None, prefix=''):
    """
    分析基准测试结果，生成统计分析和可视化
    
    参数:
        df: 结果DataFrame
        algorithms: 要分析的算法列表
        metric: 要分析的性能指标 (默认 VUS-PR)
        filter_condition: 可选的过滤条件 (函数，接受DataFrame作为参数)
        alpha: 显著性水平
        title: 图形标题
        save_dir: 保存结果的目录
        prefix: 保存文件前缀
    """
    print(f"开始分析基准测试结果，指标: {metric}")
    
    if df is None:
        print(f"没有提供数据进行分析")
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
    
    # 检查是否为多次运行的结果，需要聚合
    if 'Run' in df.columns and len(df['Run'].unique()) > 1:
        print(f"检测到多次运行的结果 ({len(df['Run'].unique())} 次), 将使用平均值进行分析")
        
        # 通过分组计算每个算法的平均指标
        df_agg = df.groupby(['Algorithm', 'Dataset']).agg({
            metric: 'mean'
        }).reset_index()
        
        # 准备数据
        eval_list = []
        for index, row in df_agg.iterrows():
            algo = row['Algorithm']
            if algo in algorithms:
                eval_list.append([algo, row['Dataset'], row[metric]])
                
        eval_df = pd.DataFrame(eval_list, columns=['classifier_name', 'dataset_name', 'accuracy'])
    else:
        # 为单次运行或已聚合的数据准备评估数据
        eval_df = prepare_benchmark_data(df, algorithms, metric_column=metric)
    
    print(f"准备好评估数据: {len(eval_df)} 条记录")
    
    # 绘制箱线图
    plt.figure(figsize=(12, 6))
    boxplot_title = f"{metric} 在不同算法上的分布" if title is None else f"{title}: {metric} 分布"
    
    # 确保只使用可用的算法列
    available_algorithms = [algo for algo in algorithms if algo in df['Algorithm'].unique()]
    if len(available_algorithms) < len(algorithms):
        print(f"警告: 只有 {len(available_algorithms)}/{len(algorithms)} 个算法在结果中可用")
    
    # 创建用于绘图的数据框
    plot_df = pd.DataFrame()
    for algo in available_algorithms:
        algo_data = df[df['Algorithm'] == algo]
        if len(algo_data) > 0 and metric in algo_data.columns:
            # 转换为数值，忽略非数值
            values = pd.to_numeric(algo_data[metric], errors='coerce').dropna()
            if len(values) > 0:
                plot_df[algo] = values
    
    if len(plot_df.columns) > 0:
        ax = plot_performance_boxplot(plot_df, plot_df.columns, metric, title=boxplot_title)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{prefix}_{metric}_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 进行统计显著性分析并绘制临界差异图
    plt.figure(figsize=(12, 6))
    cd_title = f"{metric} 算法排名临界差异图" if title is None else f"{title}: {metric} 算法排名"
    
    # 调用可视化函数
    try:
        ranking, avg_ranks = plot_critical_diagram(eval_df, available_algorithms, title=cd_title, alpha=alpha)
    except Exception as e:
        print(f"绘制临界差异图时出错: {e}")
        plt.close()
        return None, None
    
    if save_dir and ranking is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_{metric}_critical_diagram.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印排名结果
    if ranking:
        print("\n算法排名 (从高到低):")
        for i, algo in enumerate(ranking[:min(10, len(ranking))]):
            rank_val = avg_ranks[algo] if avg_ranks is not None else "N/A"
            print(f"{i+1}. {algo}: {rank_val:.3f}")
    
    return ranking, avg_ranks

def generate_comprehensive_report(results_dir, save_dir, dataset_type='Both', 
                                 metrics=None, alpha=0.05):
    """
    生成综合性报告，包括所有指标的分析
    
    参数:
        results_dir: 结果目录
        save_dir: 保存报告目录
        dataset_type: 数据集类型 ('Univariate', 'Multivariate', 或 'Both')
        metrics: 要分析的指标列表，如果为None则使用所有可用指标
        alpha: 显著性水平
    """
    # 加载结果数据
    uni_df, multi_df, stats_uni_df, stats_multi_df = load_results(results_dir, dataset_type)
    
    # 加载模型详情数据
    model_details = load_model_details(results_dir, dataset_type)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(save_dir, f"benchmark_report_{timestamp}.pdf")
    
    # 创建PDF文件
    with PdfPages(report_filename) as pdf:
        # 报告标题页
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        plt.text(0.5, 0.6, "时间序列异常检测基准测试报告", 
                fontsize=30, ha='center', fontweight='bold')
        plt.text(0.5, 0.5, f"数据集类型: {dataset_type}", 
                fontsize=20, ha='center')
        plt.text(0.5, 0.4, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                fontsize=16, ha='center')
        pdf.savefig()
        plt.close()
        
        # 确定要分析的指标
        if metrics is None:
            metrics = DEFAULT_METRIC_NAMES
        
        # 处理单变量数据集结果
        if uni_df is not None and dataset_type in ['Univariate', 'Both']:
            uni_algorithms = sorted(uni_df['Algorithm'].unique())
            
            plt.figure(figsize=(12, 4))
            plt.axis('off')
            plt.text(0.5, 0.5, "单变量数据集分析", 
                    fontsize=24, ha='center', fontweight='bold')
            pdf.savefig()
            plt.close()
            
            # 运行时间和模型大小分析
            if stats_uni_df is not None:
                analyze_runtime_and_model_size(stats_uni_df, save_dir=save_dir, prefix='univariate')
                plt.close('all')
            
            # 多次运行稳定性分析
            if 'Run' in uni_df.columns and len(uni_df['Run'].unique()) > 1:
                analyze_metrics_stability(uni_df, metrics=metrics, save_dir=save_dir, prefix='univariate')
                plt.close('all')
            
            # 分析各个指标
            for metric in metrics:
                if metric not in uni_df.columns:
                    continue
                
                title = f"单变量数据集: {metric}"
                analyze_benchmark_results(
                    uni_df, uni_algorithms, 
                    metric=metric, 
                    title=title,
                    save_dir=save_dir,
                    prefix='univariate',
                    alpha=alpha
                )
                
                # 将所有创建的图形添加到PDF
                for fig_num in plt.get_fignums():
                    plt.figure(fig_num)
                    pdf.savefig()
                plt.close('all')
        
        # 处理多变量数据集结果
        if multi_df is not None and dataset_type in ['Multivariate', 'Both']:
            multi_algorithms = sorted(multi_df['Algorithm'].unique())
            
            plt.figure(figsize=(12, 4))
            plt.axis('off')
            plt.text(0.5, 0.5, "多变量数据集分析", 
                    fontsize=24, ha='center', fontweight='bold')
            pdf.savefig()
            plt.close()
            
            # 运行时间和模型大小分析
            if stats_multi_df is not None:
                analyze_runtime_and_model_size(stats_multi_df, save_dir=save_dir, prefix='multivariate')
                plt.close('all')
            
            # 多次运行稳定性分析
            if 'Run' in multi_df.columns and len(multi_df['Run'].unique()) > 1:
                analyze_metrics_stability(multi_df, metrics=metrics, save_dir=save_dir, prefix='multivariate')
                plt.close('all')
            
            # 分析各个指标
            for metric in metrics:
                if metric not in multi_df.columns:
                    continue
                
                title = f"多变量数据集: {metric}"
                analyze_benchmark_results(
                    multi_df, multi_algorithms, 
                    metric=metric, 
                    title=title,
                    save_dir=save_dir,
                    prefix='multivariate',
                    alpha=alpha
                )
                
                # 将所有创建的图形添加到PDF
                for fig_num in plt.get_fignums():
                    plt.figure(fig_num)
                    pdf.savefig()
                plt.close('all')
        
        # 添加模型详情分析
        if model_details:
            plt.figure(figsize=(12, 4))
            plt.axis('off')
            plt.text(0.5, 0.5, "模型详细信息分析", 
                    fontsize=24, ha='center', fontweight='bold')
            pdf.savefig()
            plt.close()
            
            # 创建模型参数图表
            for algo, details_list in model_details.items():
                if not details_list:
                    continue
                
                # 提取参数信息
                params_data = []
                size_data = []
                
                for details in details_list:
                    params = details.get('total_params', None)
                    size = details.get('model_size_MB', None)
                    dataset_type = details.get('dataset_type', 'Unknown')
                    
                    if params is not None and params != 'NA':
                        try:
                            params_data.append({
                                'Algorithm': algo,
                                'Total Parameters': float(params),
                                'Dataset Type': dataset_type
                            })
                        except:
                            pass
                            
                    if size is not None and size != 'NA':
                        try:
                            size_data.append({
                                'Algorithm': algo,
                                'Model Size (MB)': float(size),
                                'Dataset Type': dataset_type
                            })
                        except:
                            pass
                
                # 如果有数据，创建图表
                if params_data:
                    plt.figure(figsize=(10, 6))
                    params_df = pd.DataFrame(params_data)
                    sns.barplot(data=params_df, x='Algorithm', y='Total Parameters', hue='Dataset Type')
                    plt.title(f"{algo} 模型参数量")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                
                if size_data:
                    plt.figure(figsize=(10, 6))
                    size_df = pd.DataFrame(size_data)
                    sns.barplot(data=size_df, x='Algorithm', y='Model Size (MB)', hue='Dataset Type')
                    plt.title(f"{algo} 模型大小 (MB)")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
    
    print(f"综合报告已生成: {report_filename}")
    return report_filename

# 默认指标名称（用于未提供metrics参数时）
DEFAULT_METRIC_NAMES = ['AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC', 
                       'Standard-F1', 'PA-F1', 'Event-based-F1', 
                       'R-based-F1', 'Affiliation-F']

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='分析基准测试结果并生成报告')
    parser.add_argument('--results_dir', type=str, required=True, help='结果目录路径')
    parser.add_argument('--save_dir', type=str, default='analysis_results', help='保存分析结果的目录')
    parser.add_argument('--dataset_type', type=str, choices=['Univariate', 'Multivariate', 'Both'], 
                       default='Both', help='要分析的数据集类型')
    parser.add_argument('--alpha', type=float, default=0.05, help='统计显著性水平')
    
    args = parser.parse_args()
    
    # 生成综合报告
    generate_comprehensive_report(
        results_dir=args.results_dir,
        save_dir=args.save_dir,
        dataset_type=args.dataset_type,
        alpha=args.alpha
    )
    
    print("分析完成！结果已保存到指定目录") 