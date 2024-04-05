#import frechetdist
import time
import sys
sys.setrecursionlimit(3000)
import torch
import similaritymeasures
import numpy as np
import neurokit2 as nk
from biosppy.signals import ecg as ecg_func
from biosppy.signals import tools as tools
import neurokit2.ppg as ppg_func
from torchmetrics.functional import pearson_corrcoef

def fid_features_to_statistics(features):
    assert torch.is_tensor(features) and features.dim() == 2
    features = features.numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return {
        'mu': mu,
        'sigma': sigma,
    }


def fid_statistics_to_metric(stat_1, stat_2):
    mu1, sigma1 = stat_1['mu'], stat_1['sigma']
    mu2, sigma2 = stat_2['mu'], stat_2['sigma']
    assert mu1.ndim == 1 and mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
    assert sigma1.ndim == 2 and sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype

    diff = mu1 - mu2
    tr_covmean = np.sum(np.sqrt(np.linalg.eigvals(sigma1.dot(sigma2)).astype('complex128')).real)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    return fid

def calculate_FD(true_ecg, fake_ecg):
        
    true_stats = fid_features_to_statistics(true_ecg.reshape(-1, 512).cpu())
    fake_stats = fid_features_to_statistics(fake_ecg.reshape(-1, 512).cpu())
    
    fd = fid_statistics_to_metric(true_stats, fake_stats) 
    
    return fd

def get_Rpeaks_ECG(filtered, sampling_rate):
    
    # segment
    rpeaks, = ecg_func.hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)

    # correct R-peak locations
    rpeaks, = ecg_func.correct_rpeaks(signal=filtered,
                             rpeaks=rpeaks,
                             sampling_rate=sampling_rate,
                             tol=0.05)

    # extract templates
    templates, rpeaks = ecg_func.extract_heartbeats(signal=filtered,
                                           rpeaks=rpeaks,
                                           sampling_rate=sampling_rate,
                                           before=0.2,
                                           after=0.4)

    rr_intervals = np.diff(rpeaks)

    return rpeaks, rr_intervals


def get_peaks_PPG(filtered, sampling_rate=128):
    
    # segment
    peaks = ppg_func.ppg_findpeaks(filtered, sampling_rate)['PPG_Peaks']
    peak_intervals = np.diff(peaks)
   
    return peaks, peak_intervals


def heartbeats_ecg(filtered, sampling_rate):
    
    rpeaks, rr_intervals = get_Rpeaks_ECG(filtered, sampling_rate)

    if rr_intervals.size != 0:
    # compute heart rate
        hr_idx, hr = tools.get_heart_rate(beats=rpeaks,
                                       sampling_rate=sampling_rate,
                                       smooth=True,
                                       size=3)

        if len(hr)==0:
            hr_idx, hr = [-1], [-1]
            
    else:
        hr_idx, hr = [-1], [-1]

    
    return hr_idx, hr


def heartbeats_ppg(filtered, sampling_rate):
    
    peaks, peaks_intervals = get_peaks_PPG(filtered, sampling_rate)

    if peaks_intervals.size != 0:
        # compute heart rate
        hr_idx, hr = tools.get_heart_rate(beats=peaks,
                                       sampling_rate=sampling_rate,
                                       smooth=True,
                                       size=3)

        if len(hr)==0:
            hr_idx, hr = [-1], [-1]
        
    else:
        hr_idx, hr = [-1], [-1]
        
    return hr_idx, hr
      
    
def ecg_bpm_array(ecg_signal, sampling_rate=128, window=4, filter=False):

    final_bpm = []
    for k in ecg_signal:
        if filter == True:
            k = nk.ecg_clean(k, sampling_rate=128, method="pantompkins1985")
        hr_idx, hr = heartbeats_ecg(k, sampling_rate)
        # print(hr)
        bpm = np.mean(hr)
        final_bpm.append(bpm)    
    return np.array(final_bpm)    

def ppg_bpm_array(ppg_signal, sampling_rate=128, window=4):
    
    final_bpm = []
    # count=0
    for k in ppg_signal:

        try:
            hr_idx, hr = heartbeats_ppg(k, sampling_rate)
            # print(count)
            bpm = np.mean(hr)
            final_bpm.append(bpm)    
            # count=count+1
        except:
            final_bpm.append(-1.0)

    return np.array(final_bpm) 

def MAE_hr(real_ecg, fake_ecg, ecg_sampling_freq=128, window_size=4):

     ######################## HR estimation from Fake ECG ######################

    real_ecg_bpm = ecg_bpm_array(real_ecg, ecg_sampling_freq, window_size)
    fake_ecg_bpm = ecg_bpm_array(fake_ecg, ecg_sampling_freq, window_size, filter=True) ## check for -1 values
    
    ## correction
    fbpm = fake_ecg_bpm[np.where(fake_ecg_bpm != -1)]
    rbpm = real_ecg_bpm[np.where(fake_ecg_bpm != -1)]

    mae_hr_ecg = np.mean(np.absolute(rbpm - fbpm))

    return mae_hr_ecg

def evaluation_pipeline(real_ecg, fake_ecg):

    rmse_score = np.sqrt(np.mean((fake_ecg - real_ecg) ** 2))
    
    mae_hr_ecg = MAE_hr(real_ecg, fake_ecg)
    
    return mae_hr_ecg, rmse_score