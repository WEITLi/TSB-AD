# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
import numpy as np # 确保导入 numpy

# --- 设置字体以支持中文 (保留以防万一，但标签将用英文) ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 手动添加项目根目录到 sys.path (如果需要导入自定义模块) ---
# script_dir = os.path.dirname(os.path.abspath(__file__)) 
# project_root = os.path.abspath(os.path.join(script_dir, '..')) 
# if project_root not in sys.path:
#     sys.path.insert(0, project_root) 
# ------------------------------------

def visualize_benchmark_results(csv_path, output_dir):
    """读取基准测试结果 CSV 并生成可视化图表（包括热力图）。""" # 更新文档字符串
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误：找不到结果文件 {csv_path}")
        return
    except Exception as e:
        print(f"读取 CSV 时出错: {e}")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 准备数据 --- 
    # 处理非数值指标（例如错误状态），将其替换为 NaN 或 0，以便绘图
    metrics_to_plot = ['AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC', 'PA-F1'] # 选择关键指标
    all_metrics = df.columns.tolist() # 获取所有列名
    # 筛选出实际存在的指标列，以便后续使用
    existing_metrics = [m for m in metrics_to_plot if m in all_metrics]
    
    if not existing_metrics:
        print("警告：在结果文件中未找到任何指定的关键指标用于绘图。")
        # 即使没有关键指标，仍然尝试绘制热力图（如果其他指标存在）
    
    # 转换所有可能的指标列为数值，错误转为 NaN
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in df.columns:
        if col not in ['Dataset', 'Type', 'Algorithm', 'Status', 'ErrorMsg', 'HP'] and col not in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 过滤掉完全失败的运行 (可选，取决于您想如何展示)
    # df_successful = df[df['Status'] == 'Success'].copy()
    # 或者保留所有运行，失败的指标将是 NaN
    df_plot = df.copy()

    # --- 2. 绘制条形图等 --- 
    sns.set_theme(style="whitegrid")

    # 图表 1: 按数据集类型和算法比较关键指标 (例如 PA-F1)
    if 'PA-F1' in existing_metrics:
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df_plot, x='Algorithm', y='PA-F1', hue='Type', palette='viridis')
        plt.title('Comparison of PA-F1 Score by Algorithm and Dataset Type', fontsize=16)
        plt.ylabel('PA-F1 Score', fontsize=14)
        plt.xlabel('Algorithm', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pa_f1_comparison.png'))
        print(f"图表已保存: {os.path.join(output_dir, 'pa_f1_comparison.png')}")
        plt.close()
    else:
        print("警告：未找到 PA-F1 指标，跳过 PA-F1 对比图绘制。")

    # 图表 2: 比较所有选定指标 (使用 FacetGrid 或分别绘制)
    # 为每个数据集类型分别绘制指标对比图
    if existing_metrics:
        for dataset_type in df_plot['Type'].unique():
            df_type = df_plot[df_plot['Type'] == dataset_type]
            
            if df_type.empty:
                continue
                
            # 将数据转换为长格式以便绘图
            df_melt = pd.melt(df_type, id_vars=['Algorithm'], value_vars=existing_metrics, var_name='Metric', value_name='Score')
            
            plt.figure(figsize=(15, 8))
            sns.barplot(data=df_melt, x='Algorithm', y='Score', hue='Metric', palette='rocket')
            plt.title(f'Algorithm Performance Comparison on {dataset_type} Dataset', fontsize=16)
            plt.ylabel('Score', fontsize=14)
            plt.xlabel('Algorithm', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(bottom=0)
            plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{dataset_type.lower()}_metrics_comparison.png'))
            print(f"图表已保存: {os.path.join(output_dir, f'{dataset_type.lower()}_metrics_comparison.png')}")
            plt.close()
    else:
        print("警告：未找到任何指定关键指标，跳过指标对比图绘制。")
        
    # 图表 3: PCA 对比 (多变量) - 逻辑保留，但注释掉，因为当前 benchmark 不支持对比
    # ... (省略 PCA 对比代码) ...
    # print("未找到用于对比的 PCA 结果 (需要 'PCA_Original' 和 'PCA_Finetuned')。")
    
    # --- 3. 创建并保存热力图 (合并自 create_heatmap.py) --- 
    print("--- 正在生成热力图 --- ")
    df_multi = df_plot[df_plot['Type'] == 'Multivariate'].copy()

    if df_multi.empty:
        print("警告：未找到多变量数据集结果，无法生成热力图。")
    else:
        # 选择用于热力图的指标 (可以与上面的 metrics_to_plot 不同)
        heatmap_metrics = ['AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC', 'PA-F1']
        existing_heatmap_metrics = [m for m in heatmap_metrics if m in df_multi.columns and df_multi[m].notna().any()]
        
        if not existing_heatmap_metrics:
            print("警告：未找到足够的多变量指标数据来生成热力图。")
        else:
            # 按算法分组并计算每个指标的平均值
            # 仅对数值列进行分组和求平均，避免非数值列的错误
            numeric_cols_multi = df_multi.select_dtypes(include=np.number).columns
            cols_to_average = [col for col in existing_heatmap_metrics if col in numeric_cols_multi]
            
            if not cols_to_average:
                 print("警告：未找到有效的数值指标列用于热力图平均值计算。")
            else:
                df_grouped = df_multi.groupby('Algorithm')[cols_to_average].mean().reset_index()

                # 创建热力图表格 (标准形式)
                plt.figure(figsize=(12, max(len(df_grouped) * 0.6, 6))) # 调整大小以适应算法数量
                heatmap_data = df_grouped.set_index('Algorithm')
                ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu', linewidths=0.5, cbar_kws={'label': 'Performance Score'})
                plt.title('Multivariate Time Series Anomaly Detection Model Performance Comparison (Heatmap)', fontsize=16)
                plt.ylabel('Model', fontsize=14)
                plt.xlabel('Evaluation Metric', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                heatmap_path = os.path.join(output_dir, 'multivariate_metrics_heatmap.png')
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                print(f"热力图已保存: {heatmap_path}")
                plt.close()

                # 创建热力图表格 (转置形式)
                plt.figure(figsize=(max(len(heatmap_data.columns) * 1.5, 8), max(len(df_grouped.Algorithm) * 0.8, 6) )) # 调整大小
                heatmap_data_transposed = heatmap_data.transpose()
                ax = sns.heatmap(heatmap_data_transposed, annot=True, fmt='.3f', cmap='YlGnBu', linewidths=0.5, cbar_kws={'label': 'Performance Score'})
                plt.title('Multivariate Time Series Anomaly Detection Evaluation Metric Comparison (Heatmap)', fontsize=16)
                plt.ylabel('Evaluation Metric', fontsize=14)
                plt.xlabel('Model', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                heatmap_transposed_path = os.path.join(output_dir, 'multivariate_metrics_heatmap_transposed.png')
                plt.savefig(heatmap_transposed_path, dpi=300, bbox_inches='tight')
                print(f"转置热力图已保存: {heatmap_transposed_path}")
                plt.close()

    # --- 4. 可视化训练时间 --- 
    print("--- 正在生成训练时间图表 --- ")
    # 确保 Time 列是数值类型，错误转为 NaN
    df_plot['Time'] = pd.to_numeric(df_plot['Time'], errors='coerce')
    # 过滤掉时间无效或运行失败的数据点以计算平均值
    df_time = df_plot[df_plot['Status'] == 'Success'].dropna(subset=['Time']).copy()

    if df_time.empty:
        print("警告：未找到有效的成功运行时间和数据，无法生成训练时间图表。")
    else:
        # 计算每种算法和类型的平均时间
        avg_time = df_time.groupby(['Type', 'Algorithm'])['Time'].mean().reset_index()

        plt.figure(figsize=(14, 8))
        # 使用条形图展示平均时间
        barplot = sns.barplot(data=avg_time, x='Algorithm', y='Time', hue='Type', palette='coolwarm')

        # 设置 Y 轴为对数刻度
        plt.yscale('log')
        # 设置 Y 轴标签和刻度格式 (可选，根据需要调整)
        # from matplotlib.ticker import ScalarFormatter
        # barplot.yaxis.set_major_formatter(ScalarFormatter())
        # 可以添加网格线
        plt.grid(True, which="both", ls="--", linewidth=0.5, axis='y') 

        plt.title('Average Training/Running Time by Algorithm and Dataset Type (Log Scale)', fontsize=16)
        plt.ylabel('Average Time (seconds, log scale)', fontsize=14)
        plt.xlabel('Algorithm', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        # 添加图例
        plt.legend(title='Dataset Type')

        # (可选) 在条形上添加文本标签显示时间
        # for container in barplot.containers:
        #     barplot.bar_label(container, fmt='%.2f', fontsize=8, padding=3)

        plt.tight_layout()
        time_plot_path = os.path.join(output_dir, 'average_running_time_comparison.png')
        plt.savefig(time_plot_path, dpi=300, bbox_inches='tight')
        print(f"训练时间图表已保存: {time_plot_path}")
        plt.close()

    print(f"可视化完成。图表保存在: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize benchmark results.')
    # 使用合并后的结果文件
    parser.add_argument('--csv_path', type=str, default='eval/benchmark/benchmark_results_all.csv', help='Path to the combined benchmark result CSV file.')
    parser.add_argument('--output_dir', type=str, default='eval/benchmark/visualizations/', help='Directory to save the generated plots.')
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_csv_path = os.path.abspath(os.path.join(script_dir, args.csv_path))
    absolute_output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    visualize_benchmark_results(absolute_csv_path, absolute_output_dir) 