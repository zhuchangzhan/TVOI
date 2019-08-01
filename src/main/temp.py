"""

This is the main program for the SPOC Catcher.

A problem with this is that PDO uses Python 2.7 and we're using 3.7

Aside from having division issues everything is fine?


Think about TVOI database


may want to move detection threshold out of the catcher


"""
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import glob
import pickle
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy import stats
import matplotlib.pyplot as plt
#from skimage.measure.tests.test_simple_metrics import cam
#plt.rcParams.update({'figure.max_open_warning': 0})

from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec

from scipy.signal import find_peaks

DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(DIR, '../..'))

import src.util.configurable as config
import src.util.gaia_query as gaia
import src.main.Lightcurve_io3 as LC_io

import src.main.FFT_Analysis as FFT
import src.main.LC_Analysis as LCA
import src.main.Period_Analysis as PA
from src.util.common_filepaths import return_all_sector_datapath,return_interested_target_datapath
from src.util.common_algorithm import segment_array,pad_array_index
from src.util.star_query import get_star_info,get_exofop_nearby

user_input = config.Configuration("../../input/configs/user_input_test.cfg")


class General_Catcher():
    
    def __init__(self):
        pass

class SPOC_Catcher():

    def __init__(self,filename,savepath,sector=0,Norm=0,detrending=721,manual_cut=[]):
    
        self.filename = filename
        self.sector = sector
        self.Normalization = Norm
        self.savepath = savepath
        self.detrending = detrending
        self.manual_cut = manual_cut
        
        # here are regions that "just works"
        # eventually I want to optimize this for physical objects
        self.min = 0
        self.slow = 0
        self.low = 6
        self.mid = 48
        self.high = 96
        self.max = 360
        
        self.slow_thres = 12
        self.low_thres  = 12
        self.mid_thres  = 12
        self.high_thres = 13
        
        self.bump_frequency = True
        self.freq_catch_cut  = 1 # truncate frequency smaller than 1
        self.Outlier_Multiplier = 0.15

    def Load_Lightcurve_Data(self):

        self.TVOI = LC_io.SPOC_TVOI(self.filename,self.savepath,self.sector,self.Normalization,self.manual_cut)
        self.TVOI.load_object_data(self.detrending)

    def Detection_Threshold(self,amplitude,rms_thres,local_pt,method,offset=0,prevmean="0"):

        if method == "Local Mean":
            
            outlier = int(local_pt*self.Outlier_Multiplier)
            seg_array = segment_array(amplitude,local_pt)
            
            bin_mean = np.sum(seg_array,1)/local_pt
            bin_rms = np.array([np.sqrt(np.sum((np.sort(local_data)[outlier:-outlier]-local_mean)**2)/(local_pt-2*outlier)) for local_mean,local_data in zip(bin_mean,seg_array)])
            bin_thres = bin_mean + bin_rms*rms_thres
        
            plot_mean = (np.ones((np.shape(seg_array)[0],local_pt))*np.array(bin_mean).reshape((-1,1))).reshape(-1)[:len(amplitude)]
            plot_thres = (np.ones((np.shape(seg_array)[0],local_pt))*np.array(bin_thres).reshape((-1,1))).reshape(-1)[:len(amplitude)]    

        elif method == "Slope Mean":
            
            outlier = int(local_pt*self.Outlier_Multiplier)
            
            low_amp_trim = np.sort(amplitude[:local_pt*2])[outlier:-outlier]
            low_mean = np.mean(low_amp_trim)
            low_rms  = np.sqrt(np.sum((low_amp_trim-low_mean)**2)/len(low_amp_trim))
            low_thres = low_mean + low_rms*rms_thres
            
            high_amp_trim = np.sort(amplitude[-local_pt*2:])[outlier:-outlier]
            high_mean = np.mean(high_amp_trim)
            high_rms  = np.sqrt(np.sum((high_amp_trim-high_mean)**2)/len(high_amp_trim))
            high_thres = high_mean + high_rms*rms_thres
            
            plot_mean = np.linspace(low_mean,high_mean,len(amplitude)+1)[:-1]
            plot_thres = np.linspace(low_thres,high_thres,len(amplitude)+1)[:-1]

        
        elif method == "Global Mean":
            
            outlier = int(len(amplitude)*self.Outlier_Multiplier)
            amplitude_trim = np.sort(amplitude)[outlier:-outlier]
            
            global_mean = np.mean(amplitude_trim)
            global_rms  = np.sqrt(np.sum((amplitude_trim-global_mean)**2)/len(amplitude_trim))
            global_thres = global_mean + global_rms*rms_thres
            
            plot_mean = np.ones(len(amplitude))*global_mean
            plot_thres = np.ones(len(amplitude))*global_thres
        
        elif method == "Boxcar Mean":
            
            Total_pt = len(amplitude)
            plot_mean = np.zeros(Total_pt)
            plot_rms  = np.zeros(Total_pt)
            
            for i in range(Total_pt):
                
                head = int(i - (local_pt)/2)+offset
                tail = int(i + (local_pt)/2)+offset
                
                if head < 0:
                    head = 0
                if tail > len(self.amplitude):
                    tail = Total_pt+offset
                local_data = self.amplitude[head:tail]
                outlier = int(len(local_data)*self.Outlier_Multiplier)
                local_mean = np.mean(local_data) 
                local_rms = np.sqrt(np.sum((np.sort(local_data)[outlier:-outlier]-local_mean)**2/(local_pt-outlier*2)))# for local_mean,local_data in zip(local_mean,seg_array)])
                plot_mean[i] = local_mean
                plot_rms[i] = local_rms
            
            
            plot_thres = plot_mean+plot_rms*rms_thres
            
        else:
            pass
        
        return plot_mean, plot_thres
            
    def Conduct_FFT_Analysis(self):
        
        self.N1 = FFT.guess_N(len(self.TVOI.time_bin),8,False)
        # TVOI.signal_bin_detrended is reserved for visualization and data folding purpose
        self.extended_time, self.extended_signal = LCA.extend_lightcurve(self.TVOI.signal_bin_cleaned,self.N1,self.Normalization)

        FFT_Result = FFT.compute_fft_general(self.extended_signal, self.TVOI.time_step, self.N1)  
        self.freq,self.sig,self.amplitude,self.power = FFT_Result


        # I want to catch things that are: 
        # 0.33-4 (3day-6hr), 4-24(6-1hr),24-96 (1hr - 15min), 96-360(15-4min)
        # slow,low,mid,high
        # consider testing a second detrending after the initial slow oscillations are caught
        
        local_pt = 2**8
        
        self.total_bin_num  = len(self.freq)
        self.slow_start_bin = int(self.total_bin_num*self.slow/self.max)
        self.low_start_bin  = int(self.total_bin_num*self.low/self.max)
        self.mid_start_bin  = int(self.total_bin_num*self.mid/self.max)
        self.high_start_bin = int(self.total_bin_num*self.high/self.max)
        
        self.slow_end_bin   = pad_array_index(self.slow_start_bin,self.low_start_bin,local_pt)
        self.low_end_bin    = pad_array_index(self.low_start_bin,self.mid_start_bin,local_pt)
        self.mid_end_bin    = pad_array_index(self.mid_start_bin,self.high_start_bin,local_pt)
        self.high_end_bin   = self.total_bin_num  # this gets special treatment because it's the end of the array, need to pad start index
        self.high_start_bin = pad_array_index(self.mid_end_bin,self.high_end_bin,local_pt,reverse=True)
        
        self.slow_mean, self.slow_thres = self.Detection_Threshold(self.amplitude[self.slow_start_bin:self.slow_end_bin],
                                                         self.slow_thres,local_pt*2,"Boxcar Mean",offset=self.slow_start_bin,prevmean=True)        
        self.low_mean, self.low_thres   = self.Detection_Threshold(self.amplitude[self.low_start_bin:self.low_end_bin],
                                                         self.low_thres,local_pt, "Boxcar Mean",offset=self.low_start_bin)
        self.mid_mean, self.mid_thres   = self.Detection_Threshold(self.amplitude[self.mid_start_bin:self.mid_end_bin],
                                                         self.mid_thres,local_pt,"Boxcar Mean",offset=self.mid_start_bin)
        self.high_mean, self.high_thres = self.Detection_Threshold(self.amplitude[self.high_start_bin:self.high_end_bin],
                                                         self.high_thres,local_pt,"Local Mean")
        
        #boxcar mean very powerful for some case and not others....
        # I need a way to use multiple detection methods and 
        # figure out a way to combine their results in future iteration of development
        # also need to calculate the amplitude of the FFT peaks
        
        self.slow_peaks,_ = find_peaks(self.amplitude[self.slow_start_bin:self.slow_end_bin],self.slow_thres)
        self.low_peaks,_ = find_peaks(self.amplitude[self.low_start_bin:self.low_end_bin],self.low_thres)
        self.mid_peaks,_ = find_peaks(self.amplitude[self.mid_start_bin:self.mid_end_bin],self.mid_thres)
        self.high_peaks,_ = find_peaks(self.amplitude[self.high_start_bin:self.high_end_bin],self.high_thres)
        
        
        self.num_slow_peaks = len(self.slow_peaks)
        self.num_low_peaks = len(self.low_peaks)
        self.num_mid_peaks = len(self.mid_peaks)
        self.num_high_peaks = len(self.high_peaks)
        
        
        self.peak_index = list(set(list(self.slow_peaks+self.slow_start_bin)+
                             list(self.low_peaks+self.low_start_bin)+
                             list(self.mid_peaks+self.mid_start_bin)+
                             list(self.high_peaks+self.high_start_bin)))
        
        self.num_peaks = len(self.peak_index)
        
        if self.num_peaks < 1:
            # future iterations should still generate simpler diagnostic plots for objects without anything found
            # this proves to be useful when someone raise certain objects to consideration 
            # also good for checking if the code id performing well
            # I will need a list of "should catch" and a list of "should not catch" for testing purpose
            # print("No FFT Peak Found")
            self.period_flag = False
            self.should_generate = False
            self.sorted_peaks = []
            return 0
            
        
        peaks = list(zip(self.amplitude[self.peak_index],self.peak_index))
        dtype = [('power', float), ('index', int)]
        self.sorted_peaks = np.sort(np.array(peaks, dtype=dtype), order='power')[::-1][:3]
        
        
        
        # Trying to refine peak detection and find principle folding frequency
        
        self.N2 = FFT.guess_N(len(self.TVOI.time_bin),128,True)
        extended_time1, extended_signal1 = LCA.extend_lightcurve(self.TVOI.signal_bin_cleaned,self.N2,self.Normalization)
        FFT_Result = FFT.compute_fft_general(extended_signal1, self.TVOI.time_step, self.N2)  
        freq1,sig1,amplitude1,power1 = FFT_Result
        
        total_bin_num1 = len(freq1)
        
        low_threshold  = int(total_bin_num1*self.slow/360)  # filter everything longer than 2 day
        high_threshold = int(total_bin_num1*24/360)   # filter everything shorter than 1 hr
        
        #target_freq = freq1[low_threshold:high_threshold]
        #target_amp  = amplitude1[low_threshold:high_threshold]
    
        predict_frequency = 1
        peak_frequencies = []
        new_peak_indexs = []
        check_param = 128 # get the amplitude of the function 
        for incident_peak in self.sorted_peaks:
            
            incident_peak_index = incident_peak[1]
            potential_peak_index = int(incident_peak_index*self.N2/self.N1)
            
            if potential_peak_index > high_threshold-check_param or potential_peak_index < low_threshold+check_param:
                continue
            
            rel_new_peak_index,_ = find_peaks(amplitude1[potential_peak_index-check_param:potential_peak_index+check_param],
                                    amplitude1[potential_peak_index-check_param])
            
            new_peak_index = rel_new_peak_index[0]+potential_peak_index-check_param
            
            frequency = freq1[new_peak_index]
            peak_frequencies.append(frequency)
            new_peak_indexs.append(new_peak_index)
        
        
        
        pfi = np.array(new_peak_indexs)
        self.period_flag = False
        if len(pfi) != 0 and self.bump_frequency:
            try:  # magic that I can't understand... presumably check if harmonic is present?
                pfi_index = [i for i, x in enumerate(pfi[1:]-int(pfi[0]/2) < 0.01*pfi[0]) if x][0]+1
                self.predict_frequency = peak_frequencies[pfi_index]
                self.period_flag = True
            except:
                pfi_index = pfi[0]
                self.predict_frequency = peak_frequencies[0]
        else:
            self.predict_frequency = self.freq[self.sorted_peaks[0][1]]
            
        
        #self.predict_frequency = peak_frequencies[1]
        # FFT amplitude doesn't matter?
        predict_frequency_amplitude = 0
        #print(self.predict_frequency)       
        
        # truncate any period longer than 1 days
        if self.predict_frequency < self.freq_catch_cut:
            self.should_generate = False
            return 0
        else:
            self.should_generate = True      
            return 1       
