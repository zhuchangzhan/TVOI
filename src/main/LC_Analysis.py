"""

Methods to analyze the lightcurve that is not related to FFT analysis

"""
import time
import os,sys
import numpy as np
from scipy.signal import find_peaks
#from astropy.convolution import convolve, Box1DKernel

DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(DIR, '../..'))

from src.util.common_algorithm import segment_array,pad_array_index

def extend_lightcurve(signal_bin,N,values=0):

    time_bin_extended = np.arange(N) # unit: 2-minute
    
    signal_bin_extended = np.concatenate([signal_bin,np.ones(N-len(signal_bin))*values])
    
    return time_bin_extended, signal_bin_extended

def calc_rms(array,pt=2**7,outlier=5,method="mean"):
    """
    
    using a local mean method
    maybe consider testing out a boxcar method?
    
    """
    
    seg_array = segment_array(array,pt)
    
    if method == "mean":
        
        bin_mean = np.sum(seg_array,1)/pt
        
        # rms is divided by the number of non zero datapoint to account for start and end of data as well as gaps and nans
        local_rms = np.array([np.sqrt(np.sum((np.sort(local_data)[outlier:-outlier]-local_mean)**2/(pt if np.count_nonzero(np.sort(local_data)[outlier:-outlier]) == 0 else np.count_nonzero(np.sort(local_data)[outlier:-outlier])))) for local_mean,local_data in zip(bin_mean,seg_array)])
        
        #threshold = local_rms*thres+bin_mean 
        
        
        num_bin2 = np.shape(seg_array)[0]
    
    
        mean_array = (np.ones((num_bin2,pt))*np.array(bin_mean).reshape((-1,1))).reshape(-1)[:len(array)] 
        rms_array  = (np.ones((num_bin2,pt))*np.array(local_rms).reshape((-1,1))).reshape(-1)[:len(array)] 
        #threshold_array = (np.ones((num_bin2,pt))*np.array(threshold).reshape((-1,1))).reshape(-1)[:len(array)] 
            
    
        return mean_array,rms_array#,threshold_array

    elif method == "global median":
        
        
        
        global_median = np.median(array)
        
        
        median_array = np.ones(len(array))*global_median
        
        rms = np.sqrt(np.sum(array-global_median)**2/(len(array)))
        
        rms_array = np.ones(len(array))*rms
        

        return median_array,rms_array


    elif method == "local median":
        
        
        bin_median = np.median(seg_array,1)
        
        # rms is divided by the number of non zero datapoint to account for start and end of data as well as gaps and nans
        local_rms = np.array([np.sqrt(np.sum((np.sort(local_data)[outlier:-outlier]-local_mean)**2/(pt if np.count_nonzero(np.sort(local_data)[outlier:-outlier]) == 0 else np.count_nonzero(np.sort(local_data)[outlier:-outlier])))) for local_mean,local_data in zip(bin_median,seg_array)])
        
        #threshold = local_rms*thres+bin_mean 
        
        
        num_bin2 = np.shape(seg_array)[0]
    
    
        median_array = (np.ones((num_bin2,pt))*np.array(bin_median).reshape((-1,1))).reshape(-1)[:len(array)] 
        rms_array  = (np.ones((num_bin2,pt))*np.array(local_rms).reshape((-1,1))).reshape(-1)[:len(array)] 
        #threshold_array = (np.ones((num_bin2,pt))*np.array(threshold).reshape((-1,1))).reshape(-1)[:len(array)] 
            
    
        return median_array,rms_array#,threshold_array


    elif method == "boxcar_mean":


        local_pt = pt
        outer = int(local_pt*0.15)
        th = outlier
        
        Total_pt = len(array)
        box_car_mean = np.zeros(Total_pt)
        box_car_rms  = np.zeros(Total_pt)
        #box_car_rms  = np.empty((Total_pt,local_pt))
        
        
        
        for i in range(Total_pt):
            
            head = int(i - (local_pt)/2)
            tail = int(i + (local_pt)/2)
            
            if head < 0:
                head = 0
            if tail > Total_pt:
                tail = Total_pt
            
            local_data = array[head:tail]
            #local_median = np.median(local_data) 
            local_mean = np.mean(local_data) 
            local_rms = np.sqrt(np.sum((np.sort(local_data)[outer:-outer]-local_mean)**2/(local_pt-outer*2)))# for local_mean,local_data in zip(local_mean,seg_array)])
            box_car_mean[i] = local_mean
            box_car_rms[i] = local_rms
            
        #threshold = box_car_mean+th*box_car_rms 
        
        return box_car_mean, box_car_rms



def find_flares(signal_bin,TESSMAG,local_bin_width = 2**7,method="local median"):
    
    #median, rms = calc_rms(signal_bin,local_bin_width, method="local median")

    mean,rms = calc_rms(signal_bin,local_bin_width,method="boxcar_mean")

    """
    if TESSMAG < 11.:
        peak_threshold = median+3*rms
        wing_threshold = median+2*rms
    elif TESSMAG < 14.:
        peak_threshold = median+4*rms
        wing_threshold = median+2*rms    
    else:
        peak_threshold = median+5*rms
        wing_threshold = median+3*rms  
    
    peak_threshold = median+3*rms
    wing_threshold = median+2*rms 
    """
    
    peak_threshold = mean+3*rms
    wing_threshold = mean+2*rms 
     
    peaks,_ = find_peaks(signal_bin,peak_threshold)
    
    trimmed_peaks = []
    failed_peaks = []
    
    for p in peaks:
        try:
            if signal_bin[p+1] > wing_threshold[p+1] and signal_bin[p+2] > wing_threshold[p+2]:  # check for 1 off spikes
                if signal_bin[p] > signal_bin[p+1] and signal_bin[p+1] > signal_bin[p+2]:
                    trimmed_peaks.append(p)
                else:
                    failed_peaks.append(p)
            else:
                failed_peaks.append(p)
        except:
            continue
    
      
    """
    for p in failed_peaks: # second chances
        if signal_bin[p] > median[p]+0.10:
            trimmed_peaks.append(p)
    
    
    # remove false peaks catching on the shoulder of the main peak
    # this also remove potential double peaks though
    final_peaks = []
    prev = 0
    for p in trimmed_peaks:
        if p-prev <5:
            continue
        else:
            final_peaks.append(p)
        prev = p
    """
    
    return trimmed_peaks,mean,rms,peak_threshold,wing_threshold

def calculate_lightcurve_variability(signal):
    
    # 101 bins from min to max of data
    bins = np.linspace(np.min(signal),np.max(signal),101)
    hist, bin_edges = np.histogram(signal,bins)





