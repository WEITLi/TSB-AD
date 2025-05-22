# TSB-AD Benchmark Analysis Tool

## Project Overview

TSB-AD (Time Series Benchmark for Anomaly Detection) is a comprehensive framework for evaluating anomaly detection algorithms on time series data. This project provides analysis tools for TSB-AD benchmark results, generating visualizations and statistics to help understand algorithm performance.

Key features include:
- Performance analysis of various anomaly detection algorithms on univariate and multivariate time series datasets
- Runtime statistics analysis
- Metrics stability analysis with mean and standard deviation in tabular format
- Result aggregation and statistical analysis for multiple experiment runs

## Environment Setup

### Requirements

This project requires Python 3.9+ and the following libraries:
tqdm
torchinfo
h5py
einops
numpy==1.26.4
pandas==2.2.3
arch>=5.3.1
hurst>=0.0.5
tslearn>=0.6.3
cython>=3.0.12
scikit-learn>=1.6.1 
scipy==1.15.2
stumpy>=1.13.0 
networkx>=3.1
transformers>=4.38.0
torch==2.3.0


### Installation Steps

1. Create and activate a conda environment:
```bash
conda create -n tsb-ad python=3.11
conda activate tsb-ad
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. If PyTorch installation fails, try:
```bash
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

4. Install the package:
```bash
pip install -e .
```

## Usage Instructions

### Running Benchmarks

1. Navigate to the `benchmark_exp` directory:
```bash
cd TSB-AD/benchmark_exp
```

2. Run the benchmark:
```bash
python run_benchmark.py --uni_dataset 451_UCR_id_149_Medical_tr_3000_1st_7175.csv --multi_dataset 178_Exathlon_id_5_Facility_tr_12538_1st_12638.csv --num_runs 5
```

Parameters:
- `--uni_dataset`: Univariate dataset filename
- `--multi_dataset`: Multivariate dataset filename
- `--num_runs`: Number of experiment repetitions for stability analysis
- `--dataset_type`: Type of dataset to analyze (Univariate/Multivariate/Both)

### Analyzing Results

Run the analysis script:
```bash
python benchmark_analysis.py --results_dir eval/benchmark/ --save_dir analysis_results --dataset_type Both
```

Parameters:
- `--results_dir`: Directory containing benchmark results
- `--save_dir`: Directory to save analysis results
- `--dataset_type`: Type of dataset to analyze (Univariate/Multivariate/Both)

## Output Results

After analysis, the following files will be generated in the `analysis_results` directory:

1. Runtime analysis charts:
   - `univariate_RunTime_s.png`
   - `multivariate_RunTime_s.png`

2. Metrics stability analysis tables:
   - `univariate_metrics_stability_combined.csv`
   - `univariate_metrics_stability_combined.png`
   - `multivariate_metrics_stability_combined.csv`
   - `multivariate_metrics_stability_combined.png`

## Key Improvements

This project enhances the original TSB-AD analysis tools with the following optimizations:

1. Created unified CSV tables for four key metrics (AUC-ROC, VUS-PR, Standard-F1, PA-F1), displaying mean and standard deviation for each

## Notes

- Ensure dataset files are in the correct directories (default: `../Datasets/TSB-AD-U/` and `../Datasets/TSB-AD-M/`)
- Analysis results are saved in the `benchmark_exp/analysis_results` directory by default
- Higher values in tables and charts indicate better performance