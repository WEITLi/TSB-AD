import numpy as np
import math
from .utils.slidingWindows import find_length_rank
import torch
import torchinfo

Unsupervise_AD_Pool = ['FFT', 'SR', 'NORMA', 'Series2Graph', 'Sub_IForest', 'IForest', 'LOF', 'Sub_LOF', 'POLY', 'MatrixProfile', 'Sub_PCA', 'PCA', 'HBOS', 
                        'Sub_HBOS', 'KNN', 'Sub_KNN','KMeansAD', 'KMeansAD_U', 'KShapeAD', 'COPOD', 'CBLOF', 'COF', 'EIF', 'RobustPCA', 'Lag_Llama', 'TimesFM', 'Chronos', 'MOMENT_ZS', 'DBSCAN']
Semisupervise_AD_Pool = ['Left_STAMPi', 'SAND', 'MCD', 'Sub_MCD', 'OCSVM', 'Sub_OCSVM', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'USAD', 'OmniAnomaly', 
                        'AnomalyTransformer', 'TimesNet', 'FITS', 'Donut', 'OFA', 'MOMENT_FT', 'M2N2', 'TCN', 'DTAAD', 'DLinear', 'BRITS', 'CSDI','SAITS']

def run_Unsupervise_AD(model_name, data, return_model_details=False, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        if return_model_details:
            results_tuple = function_to_call(data, return_model_details=True, **kwargs)
            if isinstance(results_tuple, tuple) and len(results_tuple) == 2:
                return results_tuple
            else:
                return results_tuple, None
        else:
            return function_to_call(data, return_model_details=False, **kwargs)
            
    except KeyError:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return (error_message, {"error": error_message}) if return_model_details else error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return (error_message, {"error": error_message}) if return_model_details else error_message


def run_Semisupervise_AD(model_name, data_train, data_test, return_model_details=False, **kwargs):
    print(f"--- Debug: Trying to run semi-supervised model: {model_name}")
    try:
        function_name = f'run_{model_name}'
        print(f"--- Debug: Looking for function: {function_name}")
        if function_name not in globals():
             print(f"--- Debug: Function '{function_name}' NOT FOUND in globals()!")
             run_functions = sorted([k for k in globals().keys() if k.startswith('run_')])
             print(f"--- Debug: Available run_ functions found in globals():\n{run_functions}")
             error_message = f"Function {function_name} not found."
             raise KeyError(error_message)
        
        function_to_call = globals()[function_name]
        print(f"--- Debug: Found function: {function_to_call}")
        
        if return_model_details:
            results_tuple = function_to_call(data_train, data_test, return_model_details=True, **kwargs)
            if isinstance(results_tuple, tuple) and len(results_tuple) == 2:
                return results_tuple
            else:
                print(f"Warning: {function_name} did not return a (scores, details) tuple when return_model_details=True.")
                return results_tuple, None
        else:
            return function_to_call(data_train, data_test, return_model_details=False, **kwargs)

    except KeyError as e:
        error_message = str(e)
        print(error_message)
        return (error_message, {"error": error_message}) if return_model_details else error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return (error_message, {"error": error_message}) if return_model_details else error_message

def run_FFT(data, ifft_parameters=5, local_neighbor_window=21, local_outlier_threshold=0.6, max_region_size=50, max_sign_change_distance=10):
    from .models.FFT import FFT
    clf = FFT(ifft_parameters=ifft_parameters, local_neighbor_window=local_neighbor_window, local_outlier_threshold=local_outlier_threshold, max_region_size=max_region_size, max_sign_change_distance=max_sign_change_distance)
    clf.fit(data)  
    score = clf.decision_scores_ 
    return score.ravel()

def run_Sub_IForest(data, periodicity=1, n_estimators=100, max_features=1, n_jobs=1):
    from .models.IForest import IForest
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_IForest(data, slidingWindow=100, n_estimators=100, max_features=1, return_model_details=False, n_jobs=1):
    from .models.IForest import IForest
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    
    if return_model_details:
        model_details = {
            'total_params': n_estimators,
            'trainable_params': None,
            'model_size_MB': None,
            'notes': 'For IForest, total_params refers to n_estimators. Model size in MB is not directly applicable.'
        }
        return score.ravel(), model_details
    else:
        return score.ravel()

def run_Sub_LOF(data, periodicity=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from .models.LOF import LOF
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_LOF(data, slidingWindow=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from .models.LOF import LOF
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_POLY(data, periodicity=1, power=3, n_jobs=1):
    from .models.POLY import POLY
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = POLY(power=power, window = slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_MatrixProfile(data, periodicity=1, n_jobs=1):
    from .models.MatrixProfile import MatrixProfile
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = MatrixProfile(window=slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Left_STAMPi(data_train, data):
    from .models.Left_STAMPi import Left_STAMPi
    clf = Left_STAMPi(n_init_train=len(data_train), window_size=100)
    clf.fit(data)
    score = clf.decision_function(data)
    return score.ravel()

def run_SAND(data_train, data_test, periodicity=1):
    from .models.SAND import SAND
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = SAND(pattern_length=slidingWindow, subsequence_length=4*(slidingWindow))
    clf.fit(data_test.squeeze(), online=True, overlaping_rate=int(1.5*slidingWindow), init_length=len(data_train), alpha=0.5, batch_size=max(5*(slidingWindow), int(0.1*len(data_test))))
    score = clf.decision_scores_
    return score.ravel()

def run_KShapeAD(data, periodicity=1):
    from .models.SAND import SAND
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = SAND(pattern_length=slidingWindow, subsequence_length=4*(slidingWindow))
    clf.fit(data.squeeze(), overlaping_rate=int(1.5*slidingWindow))
    score = clf.decision_scores_
    return score.ravel()

def run_Series2Graph(data, periodicity=1):
    from .models.Series2Graph import Series2Graph
    slidingWindow = find_length_rank(data, rank=periodicity)

    data = data.squeeze()
    s2g = Series2Graph(pattern_length=slidingWindow)
    s2g.fit(data)
    query_length = 2*slidingWindow
    s2g.score(query_length=query_length,dataset=data)

    score = s2g.decision_scores_
    score = np.array([score[0]]*math.ceil(query_length//2) + list(score) + [score[-1]]*(query_length//2))
    return score.ravel()

def run_Sub_PCA(data, periodicity=1, n_components=None, n_jobs=1):
    from TSB_AD.models.PCA import PCA
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_PCA(data, slidingWindow=100, n_components=None, n_jobs=1):
    from TSB_AD.models.PCA import PCA
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_NORMA(data, periodicity=1, clustering='hierarchical', n_jobs=1):
    from .models.NormA import NORMA
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = NORMA(pattern_length=slidingWindow, nm_size=3*slidingWindow, clustering=clustering)
    clf.fit(data)
    score = clf.decision_scores_
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    if len(score) > len(data):
        start = len(score) - len(data)
        score = score[start:]
    return score.ravel()

def run_Sub_HBOS(data, periodicity=1, n_bins=10, tol=0.5, n_jobs=1):
    from .models.HBOS import HBOS
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_HBOS(data, slidingWindow=1, n_bins=10, tol=0.5, n_jobs=1):
    from .models.HBOS import HBOS
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_OCSVM(data_train, data_test, kernel='rbf', nu=0.5, periodicity=1, n_jobs=1):
    from .models.OCSVM import OCSVM
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_OCSVM(data_train, data_test, kernel='rbf', nu=0.5, slidingWindow=1, n_jobs=1):
    from .models.OCSVM import OCSVM
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Sub_MCD(data_train, data_test, support_fraction=None, periodicity=1, n_jobs=1):
    from .models.MCD import MCD
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = MCD(slidingWindow=slidingWindow, support_fraction=support_fraction)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_MCD(data_train, data_test, support_fraction=None, slidingWindow=1, n_jobs=1):
    from .models.MCD import MCD
    clf = MCD(slidingWindow=slidingWindow, support_fraction=support_fraction)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Sub_KNN(data, n_neighbors=10, method='largest', periodicity=1, n_jobs=1):
    from .models.KNN import KNN
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors,method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_KNN(data, slidingWindow=1, n_neighbors=10, method='largest', n_jobs=1):
    from .models.KNN import KNN
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors, method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_KMeansAD(data, n_clusters=20, window_size=20, n_jobs=1):
    from .models.KMeansAD import KMeansAD
    clf = KMeansAD(k=n_clusters, window_size=window_size, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    return score.ravel()

def run_KMeansAD_U(data, n_clusters=20, periodicity=1,n_jobs=1):
    from .models.KMeansAD import KMeansAD
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KMeansAD(k=n_clusters, window_size=slidingWindow, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    return score.ravel()

def run_COPOD(data, n_jobs=1):
    from .models.COPOD import COPOD
    clf = COPOD(n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_CBLOF(data, n_clusters=8, alpha=0.9, n_jobs=1):
    from .models.CBLOF import CBLOF
    clf = CBLOF(n_clusters=n_clusters, alpha=alpha, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_COF(data, n_neighbors=30):
    from .models.COF import COF
    clf = COF(n_neighbors=n_neighbors)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_EIF(data, n_trees=100):
    from .models.EIF import EIF
    clf = EIF(n_trees=n_trees)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_RobustPCA(data, max_iter=1000):
    from .models.RobustPCA import RobustPCA
    clf = RobustPCA(max_iter=max_iter)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_SR(data, periodicity=1):
    from .models.SR import SR
    slidingWindow = find_length_rank(data, rank=periodicity)
    return SR(data, window_size=slidingWindow)

def run_AutoEncoder(data_train, data_test, 
                    window_size=100, hidden_neurons=[64, 32], 
                    lr=1e-3, epochs=50, batch_size=128, validation_size=0.2,
                    return_model_details=False, n_jobs=1):
    from .models.autoencoder import AutoEncoder 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    feats_train = data_train.shape[1] if len(data_train.shape) > 1 else 1
    
    clf = AutoEncoder(window_size=window_size,
                      hidden_neurons=hidden_neurons,
                      lr=lr,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_size=validation_size,
                      device=device,
                      feats=feats_train
                      )
    clf.fit(data_train)  
    score = clf.decision_function(data_test)
    
    model_details = {}
    if return_model_details:
        if hasattr(clf, 'model') and isinstance(clf.model, torch.nn.Module):
            try:
                feats_for_summary = data_test.shape[1] if len(data_test.shape) > 1 else 1
                example_input_shape = (1, window_size, feats_for_summary) 

                summary = torchinfo.summary(clf.model,
                                            input_size=example_input_shape,
                                            verbose=0,
                                            device=device,
                                            col_names = ("input_size", "output_size", "num_params", "mult_adds"))
                
                model_details['total_params'] = summary.total_params
                model_details['trainable_params'] = summary.trainable_params
                
                param_size_bytes = 0
                for param in clf.model.parameters():
                    param_size_bytes += param.nelement() * param.element_size()
                buffer_size_bytes = 0
                for buffer in clf.model.buffers():
                    buffer_size_bytes += buffer.nelement() * buffer.element_size()
                model_details['model_size_MB'] = (param_size_bytes + buffer_size_bytes) / (1024**2)
                model_details['torchinfo_summary'] = str(summary)

            except Exception as e:
                model_details['error'] = f"Could not get model details using torchinfo: {str(e)}"
                print(f"Error getting model details for AutoEncoder: {e}")
                model_details.setdefault('total_params', None)
                model_details.setdefault('trainable_params', None)
                model_details.setdefault('model_size_MB', None)
        else:
            model_details['error'] = "Model instance 'clf.model' not found or not a torch.nn.Module for AutoEncoder."
            model_details['total_params'] = None
            model_details['trainable_params'] = None
            model_details['model_size_MB'] = None
            
        return score.ravel(), model_details
    else:
        return score.ravel()

def run_CNN(data_train, data_test, window_size=100, num_channel=[32, 32, 40], lr=0.0008, n_jobs=1):
    from .models.CNN import CNN
    clf = CNN(window_size=window_size, num_channel=num_channel, feats=data_test.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_LSTMAD(data_train, data_test, window_size=100, lr=0.0008):
    from .models.LSTMAD import LSTMAD
    clf = LSTMAD(window_size=window_size, pred_len=1, lr=lr, feats=data_test.shape[1], batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_TranAD(data_train, data_test, win_size=10, lr=1e-3):
    from .models.TranAD import TranAD
    clf = TranAD(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_AnomalyTransformer(data_train, data_test, win_size=100, lr=1e-4, batch_size=128):
    from .models.AnomalyTransformer import AnomalyTransformer
    clf = AnomalyTransformer(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_OmniAnomaly(data_train, data_test, win_size=100, lr=0.002):
    from .models.OmniAnomaly import OmniAnomaly
    clf = OmniAnomaly(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_USAD(data_train, data_test, win_size=5, lr=1e-4):
    from .models.USAD import USAD
    clf = USAD(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Donut(data_train, data_test, win_size=120, lr=1e-4, batch_size=128):
    from .models.Donut import Donut
    clf = Donut(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_TimesNet(data_train, data_test, win_size=96, lr=1e-4):
    from .models.TimesNet import TimesNet
    clf = TimesNet(win_size=win_size, enc_in=data_test.shape[1], lr=lr, epochs=50)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_FITS(data_train, data_test, win_size=100, lr=1e-3):
    from .models.FITS import FITS
    clf = FITS(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_OFA(data_train, data_test, win_size=100, batch_size = 64):
    from .models.OFA import OFA
    clf = OFA(win_size=win_size, enc_in=data_test.shape[1], epochs=10, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_Lag_Llama(data, win_size=96, batch_size=64):
    from .models.Lag_Llama import Lag_Llama
    clf = Lag_Llama(win_size=win_size, input_c=data.shape[1], batch_size=batch_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Chronos(data, win_size=50, batch_size=64):
    from .models.Chronos import Chronos
    clf = Chronos(win_size=win_size, prediction_length=1, input_c=data.shape[1], model_size='base', batch_size=batch_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_TimesFM(data, win_size=96):
    from .models.TimesFM import TimesFM
    clf = TimesFM(win_size=win_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_MOMENT_ZS(data, win_size=256):
    from .models.MOMENT import MOMENT
    clf = MOMENT(win_size=win_size, input_c=data.shape[1])

    clf.zero_shot(data)
    score = clf.decision_scores_
    return score.ravel()

def run_MOMENT_FT(data_train, data_test, win_size=256):
    from .models.MOMENT import MOMENT
    clf = MOMENT(win_size=win_size, input_c=data_test.shape[1])

    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_M2N2(
        data_train, data_test, win_size=12, stride=12,
        batch_size=64, epochs=100, latent_dim=16,
        lr=1e-3, ttlr=1e-3, normalization='Detrend',
        gamma=0.99, th=0.9, valid_size=0.2, infer_mode='online'
    ):
    from .models.M2N2 import M2N2
    clf = M2N2(
        win_size=win_size, stride=stride,
        num_channels=data_test.shape[1],
        batch_size=batch_size, epochs=epochs,
        latent_dim=latent_dim,
        lr=lr, ttlr=ttlr,
        normalization=normalization,
        gamma=gamma, th=th, valid_size=valid_size,
        infer_mode=infer_mode
    )
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_TCN(data_train, data_test, window_size=100, num_channel=[25, 25, 25], kernel_size=3, dropout=0.2, lr=0.001, n_jobs=1):
    from .models.TCN import TCN
    clf = TCN(window_size=window_size, num_channel=num_channel, kernel_size=kernel_size, dropout=dropout, feats=data_test.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_DTAAD(data_train, data_test, win_size=10, lr=1e-3, batch_size=128, epochs=5):
    from TSB_AD.models.DTAAD import DTAAD
    clf = DTAAD(win_size=win_size, feats=data_test.shape[1], lr=lr, batch_size=batch_size, epochs=epochs)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_DLinear(data_train, data_test, 
                window_size=100, lr=1e-3, batch_size=128, epochs=50, 
                pred_len=1, validation_size=0.2, 
                return_model_details=False, n_jobs=1):
    from .models.DLinear import DLinear
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats_train = data_train.shape[1] if len(data_train.shape) > 1 else 1

    clf = DLinear(window_size=window_size, 
                  lr=lr, 
                  batch_size=batch_size, 
                  epochs=epochs, 
                  pred_len=pred_len, 
                  validation_size=validation_size, 
                  feats=feats_train,
                  device=device)
    clf.fit(data_train)
    score = clf.decision_function(data_test)

    model_details = {}
    if return_model_details:
        if hasattr(clf, 'model') and isinstance(clf.model, torch.nn.Module):
            try:
                feats_for_summary = data_test.shape[1] if len(data_test.shape) > 1 else 1
                example_input_shape = (1, window_size, feats_for_summary)
                
                summary = torchinfo.summary(clf.model, 
                                            input_size=example_input_shape, 
                                            verbose=0, 
                                            device=device,
                                            col_names = ("input_size", "output_size", "num_params", "mult_adds"))
                                            
                model_details['total_params'] = summary.total_params
                model_details['trainable_params'] = summary.trainable_params
                
                param_size_bytes = 0
                for param in clf.model.parameters():
                    param_size_bytes += param.nelement() * param.element_size()
                buffer_size_bytes = 0
                for buffer in clf.model.buffers():
                    buffer_size_bytes += buffer.nelement() * buffer.element_size()
                model_details['model_size_MB'] = (param_size_bytes + buffer_size_bytes) / (1024**2)
                model_details['torchinfo_summary'] = str(summary)

            except Exception as e:
                model_details['error'] = f"Could not get model details for DLinear: {str(e)}"
                print(f"Error getting model details for DLinear: {e}")
                model_details.setdefault('total_params', None)
                model_details.setdefault('trainable_params', None)
                model_details.setdefault('model_size_MB', None)
        else:
            model_details['error'] = "Model instance 'clf.model' not found or not a torch.nn.Module for DLinear."
            model_details['total_params'] = None
            model_details['trainable_params'] = None
            model_details['model_size_MB'] = None
            
        return score.ravel(), model_details
    else:
        return score.ravel()

def run_DBSCAN(data, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, return_model_details=False, n_jobs=1):
    try:
        from .models.DBSCAN import DBSCAN_AD
        clf = DBSCAN_AD(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, leaf_size=leaf_size, n_jobs=n_jobs)
        clf.fit(data)
        score = clf.decision_scores_ 
        
        if return_model_details:
            model_details = {
                'total_params': None,
                'trainable_params': None,
                'model_size_MB': None,
                'notes': f'DBSCAN parameters: eps={eps}, min_samples={min_samples}. Model size not quantifiable like NNs.'
            }
            return score.ravel(), model_details
        else:
            return score.ravel()
    except Exception as e:
        print(f"Error in run_DBSCAN: {str(e)}")
        if return_model_details:
            return str(e), {"error": f"Error in run_DBSCAN: {str(e)}"}
        else:
            raise

def run_BRITS(data_train, data_test, win_size=100, lr=1e-3, batch_size=128, epochs=50):
    from .models.BRITS import BRITS
    clf = BRITS(win_size=win_size, feats=data_test.shape[1], lr=lr, batch_size=batch_size, epochs=epochs)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_CSDI(data_train, data_test, win_size=100, lr=1e-3, batch_size=128, epochs=50):
    from .models.CSDI import CSDI
    clf = CSDI(win_size=win_size, feats=data_test.shape[1], lr=lr, batch_size=batch_size, epochs=epochs)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()

def run_SAITS(data_train, data_test, win_size=100, lr=1e-3, batch_size=128, epochs=50):
    from .models.SAITS import SAITS
    clf = SAITS(win_size=win_size, feats=data_test.shape[1], lr=lr, batch_size=batch_size, epochs=epochs)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()