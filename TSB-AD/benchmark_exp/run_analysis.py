#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time Series Anomaly Detection Benchmark Analysis Tool
This script serves as an entry point for benchmark analysis functionality
Specifically designed to analyze results produced by run_benchmark.py
"""

import os
import sys
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# --- Add project root to sys.path (ensure imports work) ---
script_dir = os.path.dirname(os.path.abspath(__file__)) # benchmark_exp directory
project_root = os.path.abspath(os.path.join(script_dir, '..')) # TSB-AD/TSB-AD directory
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"--- Debug: Added to sys.path: {project_root}")
else:
    print(f"--- Debug: {project_root} already in sys.path.")
# ------------------------------------

# Import functions from benchmark_analysis module
sys.path.append(project_root)
from benchmark_analysis import load_results, load_model_details, analyze_results_for_datasets, DEFAULT_METRIC_NAMES
from benchmark_analysis import analyze_runtime, analyze_metrics_stability_table, analyze_benchmark_results

# Configure matplotlib to use English fonts (prevent character issues)
plt.rcParams['font.sans-serif'] = ['Arial']  # Use Arial instead of SimHei
plt.rcParams['axes.unicode_minus'] = True    # Ensure minus signs display correctly

def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Run benchmark analysis and generate report')
    parser.add_argument('--results_dir', type=str, default='eval/benchmark', 
                       help='Results directory path (default: eval/benchmark)')
    parser.add_argument('--save_dir', type=str, default='analysis_results', help='Directory to save analysis results')
    parser.add_argument('--dataset_type', type=str, choices=['Univariate', 'Multivariate', 'Both'], 
                       default='Both', help='Dataset type to analyze')
    parser.add_argument('--alpha', type=float, default=0.05, help='Statistical significance level')
    parser.add_argument('--specific_metrics', action='store_true', 
                       help='Only analyze specific performance metrics, defaults to four key metrics (AUC-ROC, VUS-PR, Standard-F1, PA-F1)')
    
    args = parser.parse_args()
    
    # Print welcome message
    print(f"\n{'='*50}")
    print("Time Series Anomaly Detection Benchmark Analysis Tool v1.0")
    print(f"{'='*50}\n")
    
    # Handle relative paths
    # If relative path is provided, assume it's relative to the script directory
    if not os.path.isabs(args.results_dir):
        args.results_dir = os.path.abspath(os.path.join(script_dir, args.results_dir))
    
    if not os.path.isabs(args.save_dir):
        args.save_dir = os.path.abspath(os.path.join(script_dir, args.save_dir))
        
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' does not exist")
        print(f"Please ensure you have run run_benchmark.py and generated result files")
        print(f"Try running: python run_benchmark.py --save_dir=eval/benchmark")
        return 1
        
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Use only specified four metrics
    metrics = DEFAULT_METRIC_NAMES  # AUC-ROC, VUS-PR, Standard-F1, PA-F1
    print(f"Using metrics for analysis: {', '.join(metrics)}")
    
    print(f"Starting analysis for {args.dataset_type} dataset type")
    print(f"Results will be saved to: {args.save_dir}\n")
    
    # Configure matplotlib to not display charts, only save them
    plt.ioff()  # Turn off interactive mode
    
    # Load result data
    print("Loading result data...")
    uni_df, multi_df, stats_uni_df, stats_multi_df = load_results(
        args.results_dir, dataset_type=args.dataset_type)
        
    # Check if data was loaded successfully
    if args.dataset_type in ['Univariate', 'Both'] and uni_df is None:
        print(f"Warning: No results found for univariate datasets. Please ensure files exist in {args.results_dir}.")
        
    if args.dataset_type in ['Multivariate', 'Both'] and multi_df is None:
        print(f"Warning: No results found for multivariate datasets. Please ensure files exist in {args.results_dir}.")
    
    if (args.dataset_type == 'Both' and uni_df is None and multi_df is None) or \
       (args.dataset_type == 'Univariate' and uni_df is None) or \
       (args.dataset_type == 'Multivariate' and multi_df is None):
        print("Error: No data found for analysis")
        return 1
    
    # Analyze univariate and/or multivariate data
    if args.dataset_type in ['Univariate', 'Both'] and uni_df is not None:
        print("\nStarting univariate dataset analysis...")
        analyze_results_for_datasets(
            results_dir=args.results_dir,
            save_dir=args.save_dir,
            dataset_type='Univariate',
            metrics=metrics,
            alpha=args.alpha
        )
    
    if args.dataset_type in ['Multivariate', 'Both'] and multi_df is not None:
        print("\nStarting multivariate dataset analysis...")
        analyze_results_for_datasets(
            results_dir=args.results_dir,
            save_dir=args.save_dir,
            dataset_type='Multivariate',
            metrics=metrics,
            alpha=args.alpha
        )
    
    print(f"\n{'='*50}")
    print(f"Analysis complete! Charts saved to: {args.save_dir}")
    print(f"\nUsage tips:")
    print(f"1. First run run_benchmark.py to generate data")
    print(f"   Example: python run_benchmark.py --dataset_type Both --num_runs 2")
    print(f"2. Then run this script to analyze results")
    print(f"   Example: python run_analysis.py --results_dir eval/benchmark")
    print(f"{'='*50}")
    return 0

if __name__ == "__main__":
    exit(main()) 