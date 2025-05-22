import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime
import json

# 导入我们创建的模块
from statistical_analysis import prepare_benchmark_data, Friedman_Nemenyi
from visualization import plot_critical_diagram, plot_performance_boxplot

# 设置matplotlib英文字体支持
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用Arial字体
plt.rcParams['axes.unicode_minus'] = True    # 正常显示负号

# 只使用这四个指标进行可视化
DEFAULT_METRIC_NAMES = ['AUC-ROC', 'VUS-PR', 'Standard-F1', 'PA-F1']

def load_results(results_dir, dataset_type='Both'):
    """
    Load all result files
    
    Parameters:
        results_dir: Directory with results
        dataset_type: 'Univariate', 'Multivariate', or 'Both'
    
    Returns:
        uni_df: Univariate data
        multi_df: Multivariate data
        stats_uni_df: Univariate statistics
        stats_multi_df: Multivariate statistics
    """
    uni_df, multi_df = None, None
    stats_uni_df, stats_multi_df = None, None
    
    if dataset_type in ['Univariate', 'Both']:
        # Try to load univariate statistics
        uni_stats_path = os.path.join(results_dir, 'benchmark_stats_Univariate.csv')
        if os.path.exists(uni_stats_path):
            stats_uni_df = pd.read_csv(uni_stats_path)
            print(f"Loaded univariate statistics: {uni_stats_path}")
        
        # Try to load univariate combined results
        uni_combined_path = os.path.join(results_dir, 'benchmark_results_Univariate_combined.csv')
        if os.path.exists(uni_combined_path):
            uni_df = pd.read_csv(uni_combined_path)
            print(f"Loaded univariate combined data: {uni_combined_path}")
    
    if dataset_type in ['Multivariate', 'Both']:
        # Try to load multivariate statistics
        multi_stats_path = os.path.join(results_dir, 'benchmark_stats_Multivariate.csv')
        if os.path.exists(multi_stats_path):
            stats_multi_df = pd.read_csv(multi_stats_path)
            print(f"Loaded multivariate statistics: {multi_stats_path}")
            
        # Try to load multivariate combined results
        multi_combined_path = os.path.join(results_dir, 'benchmark_results_Multivariate_combined.csv')
        if os.path.exists(multi_combined_path):
            multi_df = pd.read_csv(multi_combined_path)
            print(f"Loaded multivariate combined data: {multi_combined_path}")
    
    # If combined data not found, try to find individual run results
    if uni_df is None and dataset_type in ['Univariate', 'Both']:
        for run_id in range(1, 10):  # Check up to 10 runs
            path = os.path.join(results_dir, f'benchmark_results_Univariate_run{run_id}.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                if uni_df is None:
                    uni_df = df
                else:
                    uni_df = pd.concat([uni_df, df], ignore_index=True)
                print(f"Loaded univariate run {run_id} data: {path}")
    
    if multi_df is None and dataset_type in ['Multivariate', 'Both']:
        for run_id in range(1, 10):  # Check up to 10 runs
            path = os.path.join(results_dir, f'benchmark_results_Multivariate_run{run_id}.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                if multi_df is None:
                    multi_df = df
                else:
                    multi_df = pd.concat([multi_df, df], ignore_index=True)
                print(f"Loaded multivariate run {run_id} data: {path}")
                
    return uni_df, multi_df, stats_uni_df, stats_multi_df

def load_model_details(results_dir, dataset_type='Both'):
    """Load model detail JSON files"""
    model_details = {}
    
    if dataset_type in ['Univariate', 'Both']:
        for run_id in range(1, 10):  # Check up to 10 runs
            path = os.path.join(results_dir, f'model_details_Univariate_run{run_id}.json')
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        # Collect information for each model across different runs
                        for algo, details in data.items():
                            if algo not in model_details:
                                model_details[algo] = []
                            details['run_id'] = run_id
                            details['dataset_type'] = 'Univariate'
                            model_details[algo].append(details)
                    print(f"Loaded univariate run {run_id} model details: {path}")
                except Exception as e:
                    print(f"Error loading {path}: {e}")
    
    if dataset_type in ['Multivariate', 'Both']:
        for run_id in range(1, 10):  # Check up to 10 runs
            path = os.path.join(results_dir, f'model_details_Multivariate_run{run_id}.json')
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        # Collect information for each model across different runs
                        for algo, details in data.items():
                            if algo not in model_details:
                                model_details[algo] = []
                            details['run_id'] = run_id
                            details['dataset_type'] = 'Multivariate'
                            model_details[algo].append(details)
                    print(f"Loaded multivariate run {run_id} model details: {path}")
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    
    return model_details

def analyze_runtime(stats_df, save_dir=None, prefix=''):
    """Analyze runtime (excluding model size)"""
    if stats_df is None or len(stats_df) == 0:
        print("No statistics data available for runtime analysis")
        return
    
    # Extract available algorithms
    algos = list(stats_df['Algorithm'].unique())
    
    # Create dataframe for plotting
    plot_data = []
    for _, row in stats_df.iterrows():
        algo = row['Algorithm']
        time_mean = row.get('Time_Mean', None)
        time_std = row.get('Time_Std', None)
        
        if pd.notna(time_mean):
            plot_data.append({
                'Algorithm': algo,
                'Metric': 'RunTime (s)',
                'Value': time_mean,
                'Std': time_std if pd.notna(time_std) else 0
            })
    
    if not plot_data:
        print("No runtime data available for plotting")
        return
        
    plot_df = pd.DataFrame(plot_data)
    
    # Create chart for RunTime only
    metric = 'RunTime (s)'
    metric_df = plot_df[plot_df['Metric'] == metric]
    if len(metric_df) == 0:
        return
        
    plt.figure(figsize=(12, 6))
    
    # Draw bar chart
    bars = plt.bar(metric_df['Algorithm'], metric_df['Value'], yerr=metric_df['Std'], 
                  capsize=5, color='skyblue', edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(metric_df['Value']),
                f'{height:.2f}', ha='center', va='bottom', rotation=0)
    
    plt.title(f"Algorithm Runtime Analysis")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric)
    plt.tight_layout()
    
    # Save chart
    if save_dir:
        save_path = os.path.join(save_dir, f"{prefix}_{metric.replace(' ', '_').replace('(', '').replace(')', '')}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved chart to: {save_path}")
    
    plt.close()  # Close chart without displaying

def analyze_metrics_stability_table(df, metrics=None, save_dir=None, prefix=''):
    """Create a combined table showing average and variance for all metrics across multiple runs"""
    if df is None or len(df) == 0:
        print("No data available for stability analysis")
        return
    
    if metrics is None:
        # Use the default four metrics
        metrics = DEFAULT_METRIC_NAMES
    
    # Extract unique algorithms and runs
    algorithms = sorted(df['Algorithm'].unique())
    runs = sorted(df['Run'].unique())
    
    if len(runs) <= 1:
        print("Only one run found, can't perform stability analysis")
        return
    
    # Create a dictionary to store all metrics for each algorithm
    combined_data = {}
    
    for algo in algorithms:
        combined_data[algo] = {}
    
    # Process each metric and store data
    for metric in metrics:
        if metric not in df.columns:
            print(f"Metric {metric} not found in data")
            continue
        
        for algo in algorithms:
            # Get metric values for this algorithm
            values = pd.to_numeric(df[df['Algorithm'] == algo][metric], errors='coerce').dropna()
            
            if len(values) > 1:  # Need at least 2 values to compute statistics
                mean_value = np.mean(values)
                std_value = np.std(values)
                
                # Store in the combined data dictionary
                combined_data[algo][f"{metric}-Mean"] = mean_value
                combined_data[algo][f"{metric}-Std"] = std_value
    
    # Create a combined DataFrame with all metrics
    combined_rows = []
    for algo, metrics_data in combined_data.items():
        if metrics_data:  # Only include algorithms with data
            row = {'Algorithm': algo}
            row.update(metrics_data)
            combined_rows.append(row)
    
    if not combined_rows:
        print("No valid stability data could be generated")
        return
    
    # Create the combined DataFrame
    combined_df = pd.DataFrame(combined_rows)
    
    # Round values for display
    numeric_columns = [col for col in combined_df.columns if col != 'Algorithm']
    combined_df[numeric_columns] = combined_df[numeric_columns].round(4)
    
    # Save as CSV
    if save_dir:
        csv_path = os.path.join(save_dir, f"{prefix}_metrics_stability_combined.csv")
        combined_df.to_csv(csv_path, index=False)
        print(f"Saved combined metrics stability table to: {csv_path}")
    
    # Create a figure for the table
    # Calculate figure size based on the number of columns and rows
    fig_width = min(16, 8 + 0.5 * len(combined_df.columns))
    fig_height = min(12, 2 + 0.5 * len(combined_df))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    # Format the column headers to be more readable
    column_labels = []
    for col in combined_df.columns:
        if col == 'Algorithm':
            column_labels.append(col)
        else:
            # Split by hyphen and format
            parts = col.split('-')
            if len(parts) == 2:
                metric, stat = parts
                column_labels.append(f"{metric}\n({stat})")
            else:
                column_labels.append(col)
    
    table = ax.table(
        cellText=combined_df.values,
        colLabels=column_labels,
        loc='center',
        cellLoc='center'
    )
    
    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Adjust column widths
    for i, width in enumerate([0.15] + [0.1] * (len(combined_df.columns) - 1)):
        for j in range(len(combined_df) + 1):
            cell = table[j, i]
            cell.set_width(width)
    
    plt.title(f"Metrics Stability Across Multiple Runs")
    plt.tight_layout()
    
    # Save table as image
    if save_dir:
        img_path = os.path.join(save_dir, f"{prefix}_metrics_stability_combined.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined metrics stability table image to: {img_path}")
        
    plt.close()

def analyze_benchmark_results(df, algorithms, metric='VUS-PR', filter_condition=None, 
                             alpha=0.05, title=None, save_dir=None, prefix=''):
    """
    Analyze benchmark results, generate statistical analysis for metrics
    
    Parameters:
        df: Results DataFrame
        algorithms: List of algorithms to analyze
        metric: Performance metric to analyze (default VUS-PR)
        filter_condition: Optional filter condition (function taking DataFrame as parameter)
        alpha: Significance level
        title: Chart title (not used anymore)
        save_dir: Directory to save results
        prefix: File prefix for saving
    """
    print(f"Starting benchmark results analysis, metric: {metric}")
    
    if df is None:
        print(f"No data provided for analysis")
        return None, None
    
    # Apply filter condition if provided
    if filter_condition is not None:
        try:
            filtered_df = filter_condition(df)
            print(f"After applying filter: {filtered_df.shape[0]} rows")
            df = filtered_df
        except Exception as e:
            print(f"Error applying filter condition: {e}")
            return None, None
    
    # Check if multiple runs, need to aggregate
    if 'Run' in df.columns and len(df['Run'].unique()) > 1:
        print(f"Detected results from multiple runs ({len(df['Run'].unique())}), will use mean for analysis")
        
        # Aggregate by grouping and calculating mean of the metric
        df_agg = df.groupby(['Algorithm', 'Dataset']).agg({
            metric: 'mean'
        }).reset_index()
        
        # Prepare data
        eval_list = []
        for index, row in df_agg.iterrows():
            algo = row['Algorithm']
            if algo in algorithms:
                eval_list.append([algo, row['Dataset'], row[metric]])
                
        eval_df = pd.DataFrame(eval_list, columns=['classifier_name', 'dataset_name', 'accuracy'])
    else:
        # Prepare evaluation data for single run or already aggregated data
        eval_df = prepare_benchmark_data(df, algorithms, metric_column=metric)
    
    print(f"Prepared evaluation data: {len(eval_df)} records")
    
    # Ensure only using available algorithm columns
    available_algorithms = [algo for algo in algorithms if algo in df['Algorithm'].unique()]
    if len(available_algorithms) < len(algorithms):
        print(f"Warning: Only {len(available_algorithms)}/{len(algorithms)} algorithms available in results")
    
    # Return empty results - no critical difference diagram or statistical analysis anymore
    return None, None

def analyze_results_for_datasets(results_dir, save_dir, dataset_type, metrics=None, alpha=0.05):
    """
    Analyze results for specific dataset type
    
    Parameters:
        results_dir: Results directory
        save_dir: Save directory
        dataset_type: Dataset type ('Univariate' or 'Multivariate')
        metrics: List of metrics to analyze
        alpha: Significance level
    """
    # Create subdirectory for dataset type results
    dataset_save_dir = os.path.join(save_dir, dataset_type.lower())
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    # Load data
    uni_df, multi_df, stats_uni_df, stats_multi_df = load_results(results_dir, dataset_type)
    
    # Determine which data and statistics to analyze
    if dataset_type == 'Univariate':
        df = uni_df
        stats_df = stats_uni_df
        prefix = 'univariate'
    else:  # 'Multivariate'
        df = multi_df
        stats_df = stats_multi_df
        prefix = 'multivariate'
    
    if df is None:
        print(f"No data found for {dataset_type} dataset")
        return
    
    algorithms = sorted(df['Algorithm'].unique())
    
    # Use default metrics or specified metrics
    if metrics is None:
        metrics = DEFAULT_METRIC_NAMES
    
    # Analyze runtime (excluding model size)
    if stats_df is not None:
        print(f"\nAnalyzing {dataset_type} dataset runtime...")
        analyze_runtime(stats_df, save_dir=dataset_save_dir, prefix=prefix)
    
    # Analyze stability (using new table format)
    if 'Run' in df.columns and len(df['Run'].unique()) > 1:
        print(f"\nAnalyzing {dataset_type} dataset stability across multiple runs...")
        analyze_metrics_stability_table(df, metrics=metrics, save_dir=dataset_save_dir, prefix=prefix)
    
    # Analyze performance metrics
    for metric in metrics:
        if metric in df.columns:
            print(f"\nAnalyzing {dataset_type} dataset {metric} metric...")
            title = f"{dataset_type} Dataset: {metric}"
            analyze_benchmark_results(
                df, algorithms,
                metric=metric,
                title=title,
                save_dir=dataset_save_dir,
                prefix=prefix,
                alpha=alpha
            )

if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze benchmark results and generate report')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to results directory')
    parser.add_argument('--save_dir', type=str, default='analysis_results', help='Directory to save analysis results')
    parser.add_argument('--dataset_type', type=str, choices=['Univariate', 'Multivariate', 'Both'], 
                       default='Both', help='Dataset type to analyze')
    parser.add_argument('--alpha', type=float, default=0.05, help='Statistical significance level')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Analyze univariate and/or multivariate data
    if args.dataset_type in ['Univariate', 'Both']:
        analyze_results_for_datasets(
            results_dir=args.results_dir, 
            save_dir=args.save_dir,
            dataset_type='Univariate',
            alpha=args.alpha
        )
    
    if args.dataset_type in ['Multivariate', 'Both']:
        analyze_results_for_datasets(
            results_dir=args.results_dir, 
            save_dir=args.save_dir,
            dataset_type='Multivariate',
            alpha=args.alpha
        )
    
    print("\nAnalysis complete! Results saved to specified directory") 