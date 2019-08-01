# -*- encoding: utf-8 -*-
"""

fft related code here



"""

import os,sys
import numpy as np
from scipy import fftpack
from scipy.signal import find_peaks
from astropy.convolution import convolve, Box1DKernel

DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(DIR, '../..'))

from src.util.common_algorithm import segment_array,pad_array_index



def guess_N(length,multiplier,over=True):
    
    cur = 1

    while 2**cur < length*multiplier:
        cur +=1
    
        
    if over:
        return 2**cur 
    else:
        return 2**(cur-1)
    
def compute_fft(TOI):
    """
    compute the fast fourier transform of the light curve
            
    need to figure out what fft and time_step is doing.
    
    """
        
    N = TOI.num_bin
    
    fft_freq   = fftpack.fftfreq(N, d=TOI.time_step)[:N // 2]            
    fft_signal = fftpack.fft(TOI.signal_bin_filtered_extended)[:N // 2] * 1 / N
    # what is the unit of this power and what does it mean
    
    # this is a rayleigh distribution, what exactly is 6 sigma correspond to
    fft_amplitude  = np.abs(fft_signal) 
    fft_power      = fft_amplitude**2
    
     
    return fft_freq,fft_signal,fft_amplitude,fft_power

def compute_fft_general(input_signal,time_step, N):
    """
    compute the fast fourier transform of the light curve
            
    need to figure out what fft and time_step is doing.
    
    """
    
    fft_freq   = fftpack.fftfreq(N, d=time_step)[:N // 2]            
    fft_signal = fftpack.fft(input_signal)[:N // 2] * 1 / N
    # what is the unit of this power and what does it mean
    
    # this is a rayleigh distribution, what exactly is 6 sigma correspond to
    fft_amplitude  = np.abs(fft_signal) 
    fft_power      = fft_amplitude**2
    
     
    return fft_freq,fft_signal,fft_amplitude,fft_power

def mdwarf_analysis(freq,amplitude,method,th):

    
    
    if method == "local_mean":
        local_pt = 2**8
        outlier = int(local_pt*0.15)
        
        
        seg_array = segment_array(amplitude,local_pt)
        
        bin_mean = np.sum(seg_array,1)/local_pt
        
        num_bin2 = np.shape(seg_array)[0]
        
    
        local_rms = np.array([np.sqrt(np.sum((np.sort(local_data)[outlier:-outlier]-local_mean)**2)/(local_pt-2*outlier)) for local_mean,local_data in zip(bin_mean,seg_array)])

        

        threshold = local_rms*th+bin_mean 
    
        plot_bin_mean = (np.ones((num_bin2,local_pt))*np.array(bin_mean).reshape((-1,1))).reshape(-1)[:len(amplitude)]
        plot_bin_thres = (np.ones((num_bin2,local_pt))*np.array(threshold).reshape((-1,1))).reshape(-1)[:len(amplitude)]    
    
    
    
        threshold = plot_bin_mean+th*plot_bin_thres
        
        
        return plot_bin_mean, threshold
        
        
    elif method == "boxcar_mean":


        local_pt = 2**11
        outlier = int(local_pt*0.15)
        
        Total_pt = len(amplitude)
        box_car_mean = np.zeros(Total_pt)
        box_car_rms  = np.zeros(Total_pt)
        #box_car_rms  = np.empty((Total_pt,local_pt))
        
        import time
        
        #seg_array = np.empty((Total_pt,local_pt))
        start = time.time()
        """
        for i in range(Total_pt):
            
            
            head = int(i - (local_pt)/2)
            tail = int(i + (local_pt)/2)
            
            if head < 0:
                head = 0
                
                seg_array[i] = np.concatenate([np.zeros(len(seg_array[1])-len(amplitude[head:tail])),amplitude[head:tail]])
                
            elif tail > Total_pt:
                tail = Total_pt
                
                seg_array[i] = np.concatenate([amplitude[head:tail],np.zeros(len(seg_array[1])-len(amplitude[head:tail]))])
                            
            else:
                seg_array[i] = amplitude[head:tail]
                
            box_car_mean[i] = np.mean(amplitude[head:tail])         
        
        print(time.time()-start)
        #num_bin2 = np.shape(seg_array)[0]
        local_rms = np.array([np.sqrt(np.sum((np.sort(local_data)[outlier:-outlier]-local_mean)**2)/(local_pt-2*outlier)) for local_mean,local_data in zip(box_car_mean,seg_array)])
        print(time.time()-start)
        threshold = box_car_mean+th*local_rms
        
        """
        for i in range(Total_pt):
            
            head = int(i - (local_pt)/2)
            tail = int(i + (local_pt)/2)
            
            if head < 0:
                head = 0
            if tail > Total_pt:
                tail = Total_pt
            
            local_data = amplitude[head:tail]
            local_mean = np.mean(local_data) 
            local_rms = np.sqrt(np.sum((np.sort(local_data)[outlier:-outlier]-local_mean)**2/(local_pt-outlier*2)))# for local_mean,local_data in zip(local_mean,seg_array)])
            box_car_mean[i] = local_mean
            box_car_rms[i] = local_rms
            
        threshold = box_car_mean+th*box_car_rms 
        
        print(time.time()-start)
        sys.exit()
        
        #
        return box_car_mean, threshold

def low_frequency_analysis(freq,amplitude,low_freq=1,mid_freq=24,high_freq=96,low_threshold=8,mid_threshold=8):
    """
    Science Case 2: Removing EBs
    
    I would want to do local bins for this
    
    Need to analyze the lightcurve for the low frequency signal first, 
    then have a way to "chop" relevent location so that high frequency analysis is possible
    
    not sure how we use the mid section...
    Also might need a section from 24-72 cycle-day... this depends on us knowing what objects we expect to find.
    
    """ 
    
    total_bin_num = len(freq)
    
    low_bin_num  = int(total_bin_num*low_freq/360)
    mid_bin_num  = int(total_bin_num*mid_freq/360)
    high_bin_num = int(total_bin_num*high_freq/360)
    
    

    #print(low_bin_num,mid_bin_num,high_bin_num)
    
    
    local_pt = int(2**7)
    
    # discribe what pad_array_index does
    lm_freq  = freq[low_bin_num:pad_array_index(low_bin_num,high_bin_num,local_pt)]
    low_freq = freq[low_bin_num:pad_array_index(low_bin_num,mid_bin_num,local_pt)]
    mid_freq = freq[mid_bin_num:pad_array_index(mid_bin_num,high_bin_num,local_pt)]
    
    lm_amp  = amplitude[low_bin_num:pad_array_index(low_bin_num,high_bin_num,local_pt)]
    low_amp = amplitude[low_bin_num:pad_array_index(low_bin_num,mid_bin_num,local_pt)]
    mid_amp = amplitude[mid_bin_num:pad_array_index(mid_bin_num,high_bin_num,local_pt)]
    
    lm_len,low_len,mid_len = len(lm_freq),len(low_amp),len(mid_amp)
    
    def calc_thres(a,pt,bin_num,th,outlier=5):
        
        
        seg_array = segment_array(a,pt)
        
        bin_mean = np.sum(seg_array,1)/pt
        
        num_bin2 = np.shape(seg_array)[0]
        

        local_rms = np.array([np.sqrt(np.sum((np.sort(local_data)[outlier:-outlier]-local_mean)**2/local_pt)) for local_mean,local_data in zip(bin_mean,seg_array)])

        threshold = local_rms*th+bin_mean 

        plot_bin_mean = (np.ones((num_bin2,local_pt))*np.array(bin_mean).reshape((-1,1))).reshape(-1)[:bin_num]
        plot_bin_thres = (np.ones((num_bin2,local_pt))*np.array(threshold).reshape((-1,1))).reshape(-1)[:bin_num] 
        
        # setting the threshold
        return seg_array, bin_mean, local_rms, threshold, plot_bin_mean, plot_bin_thres     
    
    
 
    low_array, low_mean, low_rms, low_thres, low_plot_mean, low_plot_thres = calc_thres(low_amp,local_pt,low_len,low_threshold,int(local_pt*0.15))
    
    
    low_peaks,_ = find_peaks(low_amp,low_plot_thres)
    
    if len(low_peaks) >= 1:
        low_freq_interesting = True
    else:
        low_freq_interesting = False
    

    peaks = list(zip(low_amp[low_peaks],low_peaks))
    dtype = [('power', float), ('index', int)]
    sorted_peaks = np.sort(np.array(peaks, dtype=dtype), order='power')[::-1]

    try:
        predicted_freq = low_freq[sorted_peaks[0][1]]
    except:
        predicted_freq = 1

    
    low_result = [low_freq,low_amp,low_plot_mean,low_plot_thres,low_peaks,low_freq_interesting]
    """
    plt.plot(low_freq,low_amp)
    plt.plot(low_freq,low_plot_mean,".")
    plt.plot(low_freq,low_plot_thres,".")
    plt.plot(low_freq[low_peaks],low_amp[low_peaks],"x")
    
    plt.show()
    """

    mid_array, mid_mean, mid_rms, mid_thres, mid_plot_mean, mid_plot_thres = calc_thres(mid_amp,local_pt,mid_len,mid_threshold,20)
    
    mid_peaks,_ = find_peaks(mid_amp,mid_plot_thres)
    

    if len(mid_peaks) >= 1:
        mid_freq_interesting = True
    else:
        mid_freq_interesting = False    
    
    mid_result = [mid_freq,mid_amp,mid_plot_mean,mid_plot_thres,mid_peaks,mid_freq_interesting]
    
    """
    plt.plot(mid_freq,mid_amp)
    plt.plot(mid_freq,mid_plot_mean,".")
    plt.plot(mid_freq,mid_plot_thres,".")
    
    plt.show()   
    """

    """
    lm_array, lm_mean, lm_rms, lm_thres, lm_plot_mean, lm_plot_thres = calc_thres(lm_amp,local_pt,lm_len,3,5)

    
    plt.plot(lm_freq,lm_amp)
    plt.plot(lm_freq,lm_plot_mean,".")
    plt.plot(lm_freq,lm_plot_thres,".")
    
    plt.show()
    """
    
    return low_result, mid_result,predicted_freq

def high_frequency_analysis(freq,amplitude,low_freq=60,high_freq=360,threshold=5):
    
    """
     Science Case 1: Trying to look for high frequency objects
     
     probably a global bin for 60-360 cycle/day
     
     need to make it so that this threshold is dynamic
     
    """

    total_bin_num = len(freq)
    
    high_freq_low_limit  = int(total_bin_num*low_freq/360)
    high_freq_high_limit = int(total_bin_num*high_freq/360)


    high_freq  = freq[high_freq_low_limit:high_freq_high_limit]
    high_amp  = amplitude[high_freq_low_limit:high_freq_high_limit]
    high_mean = np.mean(high_amp)
    
    high_bin = len(high_amp)
    high_rms = np.sqrt(np.sum((high_amp-high_mean)**2/high_bin))
    
    # may have a "Guess" threshold to pick out the top most peaks
    # maybe produce a distribution for the high of the peaks
    # so that I can know if it's really good peaks or junk
    # also for the top peaks, should have a metric to search for x2 and x0.5 locations
    high_thres = high_mean+high_rms*threshold
    #print(self.high_freq_mean,self.high_freq_rms)
    
    high_peaks,_ = find_peaks(high_amp,high_thres)
    
    if len(high_peaks) >= 1:
        high_freq_interesting = True
    else:
        high_freq_interesting = False     
    
    
    high_plot_mean = np.ones(high_bin)*high_mean
    high_plot_thres = np.ones(high_bin)*high_thres
    
    
    high_result = [high_freq,high_amp,high_plot_mean,high_plot_thres,high_peaks,high_freq_interesting]
    
    return high_result
    return 
    
    # need 5 sigma level
    # rayleigh distribution
    # this will take care of random peaks
    # raise the threshold so that there's no significant detection until 1-10 sources
    # be able to dynamically move the threshold
    
    """
    # peak finding, return the index for which the peaks are located
    self.peak_loc,_ = find_peaks(self.fft_high_freq_power,self.high_freq_threshold)
    
    peaks = list(zip(self.fft_high_freq_power[self.peak_loc],self.peak_loc))
    dtype = [('power', float), ('index', int)]
    self.sorted_peaks = np.sort(np.array(peaks, dtype=dtype), order='power')[::-1]
    
    # the injected signal should be removed in the final product.
    
    
    if len(peaks) >= 1:
        self.high_freq_interesting = True    
        self.predicted_freq = self.fft_high_freq[self.sorted_peaks[0][1]]
    else:
        self.high_freq_interesting = False
        self.predicted_freq = 1
    
    """

def summed_fft(TOI,fft_amplitude,times = 10):
    
    lower_limit = 100  # 100
    Slice = TOI.num_bin//2
    fft_amplitude_added = np.zeros(Slice)
    
    
    for i in range(Slice):
        if i == 0: # fix index error
            continue
        index = Slice-i

        sum_power = 0    
        split = float(index)/float(times)
        for j in range(times):
            cur_index = int(np.floor(split*(j+1.)))
            if cur_index < lower_limit:
                continue
            sum_power += fft_amplitude[cur_index]
        fft_amplitude_added[index] = sum_power  
    
    #added_peaks,_ = find_peaks(fft_amplitude_added,threshold*times)  
    
    return fft_amplitude_added

def new_summed_fft(amplitude,times = 10):
    
    lower_limit = 100  # 100
    Slice = len(amplitude)
    fft_amplitude_added = np.zeros(Slice)
    
    
    for i in range(Slice):
        if i == 0: # fix index error
            continue
        index = Slice-i

        sum_power = 0    
        split = float(index)/float(times)
        for j in range(times):
            cur_index = int(np.floor(split*(j+1.)))
            if cur_index < lower_limit:
                continue
            sum_power += amplitude[cur_index]
        fft_amplitude_added[index] = sum_power  
    
    #added_peaks,_ = find_peaks(fft_amplitude_added,threshold*times)  
    
    return fft_amplitude_added









def summed_fft_analysis(fft_freq, fft_amplitude_added, multiplier=1., remaining_peaks=20):
    
    """
    I need to be able to apply this dynamic filter 
    to other locations that need peak detection
    
    There is a problem that I need to look into the minimal distance between peaks
    I suppose this minimal distance also depend on where it is in the fft/lc
    worth thinking about.
    
    Cases with less than 5 peaks? Maybe I can implement a checker system to know
    if i need to compensate for peaks or trim peaks.
    
    What should this analysis yield? peak location?
    
    should i use bisect so that it end up being exactly 20/100 peaks?
    
    """

    sum_fft_filter = convolve(fft_amplitude_added, Box1DKernel(721))
    print("here")
    
    # add a mechanism to swap from add or subtract if the multiplier 
    # over/under estimate the number of remaining peaks.
    # if remaining_peaks is set to 20 and len(add_peaks) = 200. then pos = 1. 
    # elif len(add_peaks) = 10. pos = -1
    checker = True
    while True:
        
        added_peaks,_ = find_peaks(fft_amplitude_added,sum_fft_filter*multiplier) 
    
        if checker:
            checker = False
            if len(added_peaks) > remaining_peaks:
                threshold_accending = True
            else:
                threshold_accending = False

        if threshold_accending:
            if len(added_peaks) < remaining_peaks:
                break
            else:
                multiplier += 0.1
        else:
            if len(added_peaks) > remaining_peaks:
                break
            else:
                multiplier -= 0.1
        
        if multiplier <= 0:
            break
                
                
        print(multiplier,len(added_peaks))
        
    print(multiplier)
    
    #plt.plot(fft_freq,fft_amplitude_added)
    #plt.plot(fft_freq[added_peaks],fft_amplitude_added[added_peaks],"x")
    
    #plt.show()
    
    
    # need a way to figure out if it belongs to the same harmonics
    
    
    
    
    
    
    return sum_fft_filter, added_peaks


def new_fold_lc(x,y,period):
    
    
    if period == 1:
        return [0,0],[0,0]

def fold_lc(x,y,freq,phase=0):
    
    multiplier = 2.
    
    p = multiplier/(freq)
    if p == 1.:
        return [0,0],[0,0]
    
    #print(p)
    
    x += phase*p/multiplier
    
    new_x = []
    cur_p = p
    for i in x:
        while i > cur_p:
            cur_p += p
        new_x.append(multiplier*(((i-cur_p)/p+0.5)))
    
    return new_x,y
    
def fold_lc_analysis(x,y):
    """
    After I fold the lightcurve, I need to figure out the duty-cycle of the folded lc and know
    where to cut the data so I can do further analysis on the data.
    """
    
    
    pass
    
    
