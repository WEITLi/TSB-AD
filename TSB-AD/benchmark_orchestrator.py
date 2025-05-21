import os
import sys
import time
import random
import inspect
import numpy as np
import pandas as pd
import torch

# --- Module-level constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Attempt to import project-specific modules ---
# This assumes that when this script is run or imported,
# TSB-AD/TSB-AD/ is in the Python path or is the CWD.
try:
    from TSB_AD.models.autoencoder import AutoEncoder as AE_Class
    from TSB_AD.models.DLinear import DLinear as DLinear_Class
    from TSB_AD.models.IForest import IForest as IForest_Class
    from TSB_AD.models.PCA import PCA as PCA_Class
    from TSB_AD.models.LSTMAD import LSTMAD as LSTMAD_Class
    from TSB_AD.models.CNN import CNN as CNN_Class
    from TSB_AD.models.USAD import USAD as USAD_Class
    from TSB_AD.models.OmniAnomaly import OmniAnomaly as OmniAnomaly_Class
    from TSB_AD.models.AnomalyTransformer import AnomalyTransformer as AnomalyTransformer_Class
    from TSB_AD.models.TimesNet import TimesNet as TimesNet_Class
    from TSB_AD.models.Donut import Donut as Donut_Class
    # Add other model class imports here as needed...

    MODEL_CLASS_MAP = {
        'AutoEncoder': AE_Class,
        'DLinear': DLinear_Class,
        'IForest': IForest_Class,
        'PCA': PCA_Class,
        'LSTMAD': LSTMAD_Class,
        'CNN': CNN_Class,
        'USAD': USAD_Class,
        'OmniAnomaly': OmniAnomaly_Class,
        'AnomalyTransformer': AnomalyTransformer_Class,
        'TimesNet': TimesNet_Class,
        'Donut': Donut_Class,
        # Populate with all models you intend to get details for
        # 'SR': SR_Class, # Example for a non-HP, non-torch model if applicable
    }
    print("Benchmark Orchestrator: Successfully imported model classes.")

    from TSB_AD.model_wrapper import run_Semisupervise_AD, run_Unsupervise_AD, Semisupervise_AD_Pool, Unsupervise_AD_Pool
    from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict
    from TSB_AD.evaluation.metrics import get_metrics # For evaluate_results
    print("Benchmark Orchestrator: Successfully imported TSB_AD helper modules.")

except ImportError as e:
    print(f"ERROR: Benchmark Orchestrator failed to import TSB_AD modules: {e}")
    print("Ensure TSB-AD/TSB-AD/ is in your PYTHONPATH or is the current working directory.")
    # Initialize to prevent runtime errors if imports fail, but functionality will be limited.
    MODEL_CLASS_MAP = {}
    Semisupervise_AD_Pool = []
    Unsupervise_AD_Pool = []
    Optimal_Uni_algo_HP_dict = {}
    Optimal_Multi_algo_HP_dict = {}
    def run_Semisupervise_AD(*args, **kwargs): raise RuntimeError("TSB_AD.model_wrapper not imported")
    def run_Unsupervise_AD(*args, **kwargs): raise RuntimeError("TSB_AD.model_wrapper not imported")
    def get_metrics(*args, **kwargs): raise RuntimeError("TSB_AD.evaluation.metrics not imported")


# --- Helper Functions ---

def set_seed(seed_value=2024):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # print(f"Seed set to {seed_value}")

def _get_pytorch_model_details(model_instance):
    """Calculates size and parameters for an instantiated PyTorch model."""
    if not isinstance(model_instance, torch.nn.Module):
        return None, None, "Instance is not a torch.nn.Module"
    
    model_instance.to(DEVICE) # Ensure model is on the correct device for accurate calculation

    param_size_bytes = 0
    for param in model_instance.parameters():
        param_size_bytes += param.nelement() * param.element_size()
    
    buffer_size_bytes = 0
    for buffer in model_instance.buffers():
        buffer_size_bytes += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size_bytes + buffer_size_bytes) / (1024**2)
    num_total_params = sum(p.numel() for p in model_instance.parameters())
    num_trainable_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
    
    details = {
        "total_params": num_total_params,
        "trainable_params": num_trainable_params,
        "model_size_MB": size_mb,
        "device": str(DEVICE)
    }
    return details

def get_model_instance_details(model_name, model_instance, hps=None):
    """
    Gets details (params, size) for a given instantiated model.
    Handles PyTorch and provides placeholders for other model types.
    """
    details = {
        "model_name": model_name,
        "total_params": None,
        "trainable_params": None,
        "model_size_MB": None,
        "notes": "",
        "error": None
    }
    hps = hps or {}

    if isinstance(model_instance, torch.nn.Module):
        try:
            pt_details = _get_pytorch_model_details(model_instance)
            if isinstance(pt_details, dict): # Check if successful
                 details.update(pt_details)
            else: # pt_details might be (None, None, error_msg)
                 details["error"] = pt_details
        except Exception as e:
            details["error"] = f"Failed to get PyTorch model details: {str(e)}"
    elif model_name == 'IForest': # Example for a non-PyTorch model
        details["total_params"] = hps.get('n_estimators', 'N/A') # n_estimators from HP
        details["notes"] = "For IForest, total_params refers to n_estimators. Size in MB not applicable."
    elif model_name == 'PCA':
        details["total_params"] = hps.get('n_components', 'N/A')
        details["notes"] = "For PCA, total_params refers to n_components. Size in MB not applicable."
    # Add more 'elif model_name == ...' blocks for other specific non-PyTorch models
    else:
        details["notes"] = "Model type not specifically handled for detailed parameter counting beyond PyTorch."
        if hasattr(model_instance, 'get_params'): # For scikit-learn like models
            try:
                details['sklearn_params'] = model_instance.get_params()
            except:
                pass
    return details

# --- Main Experiment Runner ---

def run_benchmark_experiment(AD_Name, data_train, data_test, 
                             is_multivariate, dataset_name, 
                             n_runs=2, base_seed=2024):
    """
    Runs a benchmark experiment for a single algorithm over multiple runs.
    It re-instantiates the model to get details, avoiding modification to model_wrapper.py.
    """
    all_run_scores = []
    all_run_inference_times = []
    all_run_model_details = []

    hp_dict_source = Optimal_Multi_algo_HP_dict if is_multivariate else Optimal_Uni_algo_HP_dict
    
    if AD_Name not in hp_dict_source:
        print(f"WARNING: Hyperparameters for {AD_Name} (multivariate={is_multivariate}) not found. Skipping experiment.")
        return None, None, None
    
    optimal_hps = hp_dict_source[AD_Name]

    print(f"\n--- Starting Experiment ---")
    print(f"Algorithm: {AD_Name}, Dataset: {dataset_name}, Multivariate: {is_multivariate}, Runs: {n_runs}")
    print(f"Using HPs: {optimal_hps}")

    for i_run in range(n_runs):
        current_seed = base_seed + i_run
        set_seed(current_seed)
        print(f"  Run {i_run + 1}/{n_runs} (Seed: {current_seed})...")

        # 1. Get anomaly scores using existing model_wrapper
        scores = None
        inference_time = -1
        run_successful = False
        
        start_time = time.time()
        try:
            if AD_Name in Semisupervise_AD_Pool:
                if data_train is None:
                    raise ValueError(f"data_train is required for semi-supervised model {AD_Name}")
                scores = run_Semisupervise_AD(AD_Name, data_train, data_test, **optimal_hps)
            elif AD_Name in Unsupervise_AD_Pool:
                scores = run_Unsupervise_AD(AD_Name, data_test, **optimal_hps)
            else:
                raise ValueError(f"Algorithm {AD_Name} not found in supervised/unsupervised pools.")
            
            if isinstance(scores, np.ndarray):
                run_successful = True
            else: # model_wrapper might return an error string
                print(f"    WARNING: Model {AD_Name} run {i_run+1} did not return a numpy array. Output: {type(scores)}")
                scores = None # Ensure scores is None if not a valid array
        except Exception as e:
            print(f"    ERROR: Running model {AD_Name} (run {i_run+1}) via model_wrapper failed: {e}")
            scores = None
        inference_time = time.time() - start_time

        all_run_scores.append(scores) # Append scores array or None
        all_run_inference_times.append(inference_time if run_successful else -1)
        
        if run_successful:
             print(f"    Run {i_run+1} scores obtained. Inference Time: {inference_time:.3f}s")
        else:
             print(f"    Run {i_run+1} failed to obtain scores.")


        # 2. Get model details by re-instantiating the model
        model_details_dict = {"model_name": AD_Name, "error": "Details not obtained"}
        if AD_Name in MODEL_CLASS_MAP:
            ModelClass = MODEL_CLASS_MAP[AD_Name]
            model_init_params = {**optimal_hps} # Start with HPs

            try:
                # Prepare additional params like 'feats', 'device' if ModelClass expects them
                # This part is crucial and model-specific.
                current_data_for_feats = data_train if AD_Name in Semisupervise_AD_Pool and data_train is not None else data_test
                if current_data_for_feats is not None:
                    feats = current_data_for_feats.shape[1] if len(current_data_for_feats.shape) > 1 else 1
                    
                    # Check common feature parameter names; this might need refinement
                    sig_params = inspect.signature(ModelClass.__init__).parameters
                    if 'feats' in sig_params and 'feats' not in model_init_params : model_init_params['feats'] = feats
                    elif 'input_c' in sig_params and 'input_c' not in model_init_params: model_init_params['input_c'] = feats
                    elif 'features' in sig_params and 'features' not in model_init_params: model_init_params['features'] = feats
                    elif 'd_feat' in sig_params and 'd_feat' not in model_init_params: model_init_params['d_feat'] = feats
                
                # Add device for PyTorch models (heuristically or by checking base class)
                # A more robust way would be to know which models are PyTorch based.
                if 'torch.nn.modules' in str(ModelClass.__mro__): # Check if inherits from torch.nn.Module
                    model_init_params['device'] = DEVICE
                
                # Filter params to only those accepted by ModelClass.__init__
                sig = inspect.signature(ModelClass.__init__)
                constructor_params = {p_name for p_name in sig.parameters if p_name != 'self'}
                final_init_params = {k: v for k, v in model_init_params.items() if k in constructor_params}
                
                # Check for missing essential params if ModelClass doesn't use **kwargs in __init__
                # This is complex; for now, we assume HPs + inferred feats/device are mostly sufficient or model handles missing with defaults.

                # print(f"    Debug: Re-instantiating {AD_Name} for details with params: {final_init_params}")
                temp_model_instance = ModelClass(**final_init_params)
                model_details_dict = get_model_instance_details(AD_Name, temp_model_instance, optimal_hps)
                print(f"    Run {i_run+1} model details: Params={model_details_dict.get('total_params', 'N/A')}, SizeMB={model_details_dict.get('model_size_MB', 'N/A')}")

            except Exception as e:
                print(f"    ERROR: Re-instantiating {AD_Name} or getting details for run {i_run+1} failed: {e}")
                model_details_dict["error"] = f"Failed to instantiate or get details: {str(e)}"
        else:
            print(f"    WARNING: Model class for {AD_Name} not found in MODEL_CLASS_MAP. Cannot get details.")
            model_details_dict["error"] = "Model class not in map."
            
        all_run_model_details.append(model_details_dict)

    return all_run_scores, all_run_inference_times, all_run_model_details

# --- Evaluation Function ---

def evaluate_run_results(list_of_scores, labels_list, sliding_window_list, metric_names=None):
    """
    Evaluates results from multiple runs (list of score arrays).
    labels_list and sliding_window_list should correspond to each score array if they vary,
    or can be single values if they are constant across all scores.
    """
    if metric_names is None: # Default TSB-AD metrics
        metric_names = ['AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC', 
                        'Standard-F1', 'PA-F1', 'Event-based-F1', 
                        'R-based-F1', 'Affiliation-F']

    all_run_metrics = []

    if not isinstance(labels_list, list): labels_list = [labels_list] * len(list_of_scores)
    if not isinstance(sliding_window_list, list): sliding_window_list = [sliding_window_list] * len(list_of_scores)

    for i, scores_arr in enumerate(list_of_scores):
        if scores_arr is None or not isinstance(scores_arr, np.ndarray):
            print(f"  Run {i+1} had no valid scores. Skipping evaluation for this run.")
            # Append a dictionary of NaNs or error indicators for this run
            all_run_metrics.append({metric: np.nan for metric in metric_names})
            continue
        
        current_labels = labels_list[i % len(labels_list)] # Cycle through if lists are shorter
        current_sw = sliding_window_list[i % len(sliding_window_list)]

        try:
            eval_metrics = get_metrics(scores_arr, current_labels, slidingWindow=current_sw)
            all_run_metrics.append(eval_metrics)
        except Exception as e:
            print(f"  ERROR: Evaluating metrics for run {i+1} failed: {e}")
            all_run_metrics.append({metric: np.nan for metric in metric_names}) # Append NaNs

    if not all_run_metrics:
        return pd.DataFrame(), pd.DataFrame() # Empty DataFrames

    df_all_metrics = pd.DataFrame(all_run_metrics)
    
    # Calculate mean and std, handling potential all-NaN columns if all runs failed for a metric
    mean_metrics = df_all_metrics.mean(numeric_only=True)
    std_metrics = df_all_metrics.std(numeric_only=True)
    
    # For non-numeric (e.g., if 'EvalError' was stored), describe might be better
    # Or convert specific error strings to NaN before mean/std

    return mean_metrics, std_metrics, df_all_metrics

# --- Example of how this module might be used from another script (e.g., your Colab notebook) ---
if __name__ == "__main__":
    print("\n--- Running benchmark_orchestrator.py as main (example usage) ---")
    # This block is for testing this module directly.
    # In Colab, you would import functions from this module.

    # Create dummy data for testing
    set_seed(42)
    dummy_data_train = np.random.rand(1000, 5)
    dummy_data_test = np.random.rand(2000, 5)
    dummy_labels = np.random.randint(0, 2, 2000)
    dummy_sliding_window = 50
    dummy_dataset_name = "DummyMultiDataset"

    # Test with a model known to be in MODEL_CLASS_MAP and HP_list
    # Ensure 'AutoEncoder' is in Optimal_Multi_algo_HP_dict for this to run
    if 'AutoEncoder' in Optimal_Multi_algo_HP_dict and 'AutoEncoder' in MODEL_CLASS_MAP:
        print("\nTesting with AutoEncoder (Multivariate)...")
        ae_scores, ae_times, ae_details = run_benchmark_experiment(
            AD_Name='AutoEncoder',
            data_train=dummy_data_train,
            data_test=dummy_data_test,
            is_multivariate=True,
            dataset_name=dummy_dataset_name,
            n_runs=2,
            base_seed=2024
        )

        if ae_scores and any(s is not None for s in ae_scores):
            mean_m, std_m, all_m_df = evaluate_run_results(ae_scores, dummy_labels, dummy_sliding_window)
            print("\nAutoEncoder Mean Metrics:\n", mean_m)
            print("\nAutoEncoder Std Dev Metrics:\n", std_m)
            print("\nAutoEncoder All Run Metrics DF:\n", all_m_df.head())
            print("\nAutoEncoder Inference Times:", ae_times)
            print("\nAutoEncoder Model Details per run:")
            for i, detail in enumerate(ae_details):
                print(f"  Run {i+1}: {detail}")
        else:
            print("AutoEncoder experiment did not yield scores.")
    else:
        print("Skipping AutoEncoder test: Not found in HP_list or MODEL_CLASS_MAP or TSB_AD modules not loaded.")

    # Test with a different model, e.g., IForest (Unsupervised)
    # Ensure 'IForest' is in Optimal_Uni_algo_HP_dict
    if 'IForest' in Optimal_Uni_algo_HP_dict and 'IForest' in MODEL_CLASS_MAP:
        print("\nTesting with IForest (Univariate - using first feature of dummy data)...")
        dummy_data_test_uni = dummy_data_test[:, 0].reshape(-1, 1) # Make it 2D for IForest wrapper
        if_scores, if_times, if_details = run_benchmark_experiment(
            AD_Name='IForest',
            data_train=None, # Unsupervised
            data_test=dummy_data_test_uni,
            is_multivariate=False, # Assuming IForest HP is in Uni dict
            dataset_name="DummyUniDataset",
            n_runs=2,
            base_seed=2030
        )
        if if_scores and any(s is not None for s in if_scores):
            dummy_labels_uni = np.random.randint(0, 2, len(dummy_data_test_uni))
            dummy_sw_uni = 30
            mean_m_if, std_m_if, all_m_df_if = evaluate_run_results(if_scores, dummy_labels_uni, dummy_sw_uni)
            print("\nIForest Mean Metrics:\n", mean_m_if)
            print("\nIForest Model Details per run:")
            for i, detail in enumerate(if_details):
                print(f"  Run {i+1}: {detail}")
        else:
            print("IForest experiment did not yield scores.")

    else:
        print("Skipping IForest test: Not found in HP_list or MODEL_CLASS_MAP or TSB_AD modules not loaded.")
    print("\n--- benchmark_orchestrator.py example usage finished ---") 