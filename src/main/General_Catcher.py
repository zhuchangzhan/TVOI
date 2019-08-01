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
import csv
import sys
import time
import glob
import pickle
import numpy as np
import pandas as pd
import peakutils
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
#import src.main.Lightcurve_io3 as LC_io
import src.main.Lightcurve_io4 as LC_io2
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

    def Load_Lightcurve_Data(self,user=False):

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

    def Generate_1Pager(self,deploy=False,forced=False):
        
        if not forced and (len(self.sorted_peaks) == 0 or not self.should_generate):
            return
        #print(self.should_generate)
        
        
        TVOI = self.TVOI
        
        try:
            Tmag  = float(TVOI.object["TESSMAG"])
        except:
            Tmag  = 0
        
        try:
            Rstar = float(TVOI.object["RADIUS"])
        except:
            Rstar = 0
            
        try:
            Temp  = float(TVOI.object["TEFF"])
        except:
            Temp = 0
        #Logg  = float(TVOI.object["LOGG"])
        #mH    = float(TVOI.object["MH"])
        try:
            ra    = float(TVOI.object["RA_OBJ"])
        except:
            ra = 0
        try:
            dec   = float(TVOI.object["DEC_OBJ"])
        except:
            dec = 0
        try:
            pmra  = float(TVOI.object["PMRA"])
        except:
            pmra = 0
        try:
            pmdec = float(TVOI.object["PMDEC"])
        except:
            pmdec = 0
        try:
            pmtot = float(TVOI.object["PMTOTAL"])
        except:
            pmtot = 0
        
        try:
            target_gaia_data = gaia.gaia_query(ra,dec,0.005)
        except:
            target_gaia_data = gaia.gaia_query(ra,dec,0.015)
            
        Gmag = min(target_gaia_data["Gmag"])
        parallex = max(target_gaia_data["RPlx"])/1000 #miliarcsec - arcsec
        distance = 1./parallex
        
        #print(distance)
        if distance == np.nan or str(distance) == "nan" or str(distance) == "--":
            distance = 0
        if Rstar == np.nan or str(Rstar) == "nan":
            Rstar = 0
            
            
        
        gaia_data = gaia.gaia_query(ra,dec,0.016)
        
        #if self.classification == None: 
        #    classification = "/"
        #else:
        
        #print(self.classification)
        try:
            classification = "/".join(self.classification)
        except:
            classification = "N"
        
        fig = plt.figure(figsize=(18,9),constrained_layout=True)#figsize=(18,9))#constrained_layout=True)
        gs = GridSpec(4,7, figure=fig)



        main_LC = fig.add_subplot(gs[0, 1:])
        main_LC.set_title("""TIC ID: %s (s:%s c:%s CCD:%s)                 T$_{mag}$: %.3f G$_{mag}$: %.3f T$_{eff}$: %sK R$_S$: %sR$\odot$ D$_S$: %.2fpc
                           \nFFT peaks: %s Flares: %s Classification:%s        Ra: %.5f Dec: %.5f pmra: %.4f pmdec: %.4f pmtot: %.4f"""%(
                        TVOI.TIC_ID,TVOI.sector,TVOI.camera,TVOI.ccd,
                        Tmag,Gmag,Temp,Rstar,distance,
                        self.num_peaks,TVOI.num_flares,classification,
                        ra,dec,pmra,pmdec,pmtot))
    
        
        
        main_LC.plot(TVOI.time_raw_uncut,TVOI.signal_calibrated_uncut,".",color="0.3",markersize=2,label="calibrated")
        main_LC.plot(TVOI.time_bin*TVOI.bin2day+TVOI.start_time,TVOI.signal_bin_cleaned,label="binned")
        main_Max = np.max(TVOI.signal_bin_cleaned)*1.2
        main_Min = np.min(TVOI.signal_bin_cleaned)*1.2
        main_LC.set_ylim(main_Min,main_Max)
        
        
        
        zoomed_LC = fig.add_subplot(gs[0, 0])
        zoomed_LC.set_title("Zoomed LC (~2d)",fontsize=10)
        zoomed_low = 2048*2
        zoomed_high = 2048*2+1024
        zoomed_LC.plot(TVOI.time_calibrated[zoomed_low:zoomed_high]+TVOI.start_time,TVOI.signal_calibrated[zoomed_low:zoomed_high],".",markersize="2",color="0.3")
            
        
        

        
        
        raw_LC = fig.add_subplot(gs[1, 0])
        raw_LC.set_title("Raw Lightcurve",fontsize=10)
        raw_LC.plot(TVOI.time_raw_uncut,TVOI.signal_raw_uncut,".",markersize=2,color="0.3")
        raw_LC.xaxis.set_visible(False)
        raw_Max = np.max(TVOI.signal_raw)*1.05
        raw_Min = np.min(TVOI.signal_raw)*0.95
        raw_LC.set_ylim(raw_Min,raw_Max)
        
        centx_LC = fig.add_subplot(gs[2, 0])
        centx_LC.set_title("Centroid X/Y",fontsize=10)
        centx_LC.plot(TVOI.time_raw,TVOI.centdx-np.mean(TVOI.centdx),".",markersize=2,label="X")
        centx_LC.plot(TVOI.time_raw,TVOI.centdy-np.mean(TVOI.centdy),".",markersize=2,label="Y")
        centx_LC.xaxis.set_visible(False)
        centx_LC.legend()
        
        sonogram = fig.add_subplot(gs[3, 0])
        sonogram.set_title("Sonogram",fontsize=10)
        
        f, t, Sxx = scipy_signal.spectrogram(TVOI.signal_bin_cleaned,720)
        sonogram.pcolormesh(t, f, np.sqrt(Sxx))
        sonogram.set_ylim(0,48)
        #sonogram.set_ylabel('Frequency [Hz]')
        #sonogram.set_xlabel('Time [sec]')
        
        zoom_FFT    = fig.add_subplot(gs[1, 1:5])
        zoom_FFT.set_title("FFT ( 0 - 96 cycle/day)",fontsize=10)
        zoom_FFT.plot(self.freq,self.amplitude)
        zoom_FFT.plot(self.freq[self.peak_index],self.amplitude[self.peak_index],"*",color="b")
        try:
            zoom_FFT.plot(self.freq[self.sorted_peaks["index"]],self.amplitude[self.sorted_peaks["index"]],"*",color="r")
        except:
            pass
        zoom_FFT.set_xlim(0,100)


        
        log_FFT     = fig.add_subplot(gs[1, 5:7])
        log_FFT.set_title("Log FFT",fontsize=10)
        log_FFT.plot(self.freq,self.amplitude,label="Amp")
        log_FFT.plot(self.freq[self.slow_start_bin:self.slow_end_bin],self.slow_mean,color="k",label="Mean")
        log_FFT.plot(self.freq[self.slow_start_bin:self.slow_end_bin],self.slow_thres,color="r",label="Thres")
        log_FFT.plot(self.freq[self.low_start_bin:self.low_end_bin],self.low_mean,color="k")
        log_FFT.plot(self.freq[self.low_start_bin:self.low_end_bin],self.low_thres,color="r")
        log_FFT.plot(self.freq[self.mid_start_bin:self.mid_end_bin],self.mid_mean,color="k")
        log_FFT.plot(self.freq[self.mid_start_bin:self.mid_end_bin],self.mid_thres,color="r")   
        log_FFT.plot(self.freq[self.high_start_bin:self.high_end_bin],self.high_mean,color="k")
        log_FFT.plot(self.freq[self.high_start_bin:self.high_end_bin],self.high_thres,color="r")
        try:
            log_FFT.plot(self.freq[self.peak_index],self.amplitude[self.peak_index],"*",color="b")
            log_FFT.plot(self.freq[self.sorted_peaks["index"]],self.amplitude[self.sorted_peaks["index"]],"*",color="r")
        except:
            pass
        log_FFT.set_xlim(1,360)
        log_FFT.set_yscale("log")
        log_FFT.set_xscale("log")   
        log_FFT.legend()     
        
        slow_FFT    = fig.add_subplot(gs[2, 1])
        slow_FFT.set_title("FFT ( 0 - 6 cycle/day)",fontsize=10)
        slow_FFT.plot(self.freq,self.amplitude)
        slow_FFT.plot(self.freq[self.slow_start_bin:self.slow_end_bin],self.slow_mean,color="k")
        slow_FFT.plot(self.freq[self.slow_start_bin:self.slow_end_bin],self.slow_thres,color="r")
        slow_max = max(max(self.amplitude[(self.freq<=6)]),max(self.slow_thres))*1.2
        slow_FFT.set_xlim(0,6)     
        slow_FFT.set_ylim(0,slow_max)   
        slow_FFT.yaxis.set_visible(False) 
        
        low_FFT    = fig.add_subplot(gs[2, 2])
        low_FFT.set_title("FFT ( 6 - 48 cycle/day)",fontsize=10)
        low_FFT.plot(self.freq,self.amplitude)
        low_FFT.plot(self.freq[self.low_start_bin:self.low_end_bin],self.low_mean,color="k")
        low_FFT.plot(self.freq[self.low_start_bin:self.low_end_bin],self.low_thres,color="r")
        low_max = max(max(self.amplitude[(self.freq>=6) &(self.freq<=48)]),max(self.low_thres))*1.2
        low_FFT.set_xlim(6,48)      
        low_FFT.set_ylim(0,low_max)    
        low_FFT.yaxis.set_visible(False) 
        
        mid_FFT     = fig.add_subplot(gs[2, 3])
        mid_FFT.set_title("FFT ( 48 - 96 cycle/day)",fontsize=10)
        mid_FFT.plot(self.freq,self.amplitude)
        mid_FFT.plot(self.freq[self.mid_start_bin:self.mid_end_bin],self.mid_mean,color="k")
        mid_FFT.plot(self.freq[self.mid_start_bin:self.mid_end_bin],self.mid_thres,color="r") 
        mid_max = max(max(self.amplitude[(self.freq>=48) &(self.freq<=96)]),max(self.mid_thres))*1.2
        mid_FFT.set_xlim(48,96)  
        mid_FFT.set_ylim(0,mid_max)         
        mid_FFT.yaxis.set_visible(False) 
 
        high_FFT     = fig.add_subplot(gs[2, 4])
        high_FFT.set_title("FFT ( 96 - 360 cycle/day)",fontsize=10)
        high_FFT.plot(self.freq,self.amplitude)
        high_FFT.plot(self.freq[self.high_start_bin:self.high_end_bin],self.high_mean,color="k")
        high_FFT.plot(self.freq[self.high_start_bin:self.high_end_bin],self.high_thres,color="r")
        high_max = max(max(self.amplitude[(self.freq>=96)&(self.freq<=360)]),max(self.high_thres))*1.2
        high_FFT.set_xlim(96,360)  
        high_FFT.set_ylim(0,high_max)   
        high_FFT.yaxis.set_visible(False) 
        
        summed_FFT  = fig.add_subplot(gs[2, 5:7])
        summed_FFT.set_title("Summed FFT",fontsize=10)
        FFT_Summed = FFT.new_summed_fft(self.amplitude)
        summed_FFT.plot(self.freq,FFT_Summed)      
          

        fold_LC     = fig.add_subplot(gs[3, 1:4])
        if len(self.sorted_peaks)>=1:
            if self.period_flag:
                fold_LC.set_title("Detrended LC Folded @ P = %.6f days (%.3f hr)"%(1/self.predict_frequency,24/self.predict_frequency),fontsize=10,color="r")
            else:
                fold_LC.set_title("Detrended LC Folded @ P = %.6f days (%.3f hr)"%(1/self.predict_frequency,24/self.predict_frequency),fontsize=10)
        else:
            fold_LC.set_title("Detrended LC Not Folded",fontsize=10)
                    
            
             
        dp = int(len(self.TVOI.time_raw)/4)
        time_bin = self.TVOI.time_bin*self.TVOI.bin2day+self.TVOI.start_time
        signal_bin = self.TVOI.signal_bin_cleaned.copy()
        signal_bin_bin = self.TVOI.signal_bin_cleaned.copy()
        signal_bin[signal_bin==0] = np.nan
        
        if len(self.sorted_peaks) >= 1:
            foldx0,foldy0 = FFT.fold_lc(time_bin,signal_bin_bin,self.predict_frequency)
            foldx1,foldy1 = FFT.fold_lc(time_bin[:dp],signal_bin[:dp],self.predict_frequency)
            foldx2,foldy2 = FFT.fold_lc(time_bin[dp:2*dp],signal_bin[dp:2*dp],self.predict_frequency)
            foldx3,foldy3 = FFT.fold_lc(time_bin[2*dp:3*dp],signal_bin[2*dp:3*dp],self.predict_frequency)
            foldx4,foldy4 = FFT.fold_lc(time_bin[3*dp:],signal_bin[3*dp:],self.predict_frequency)        
            
            bin_means, bin_edges, binnumber = stats.binned_statistic(foldx0,foldy0, 
                                                                     statistic='mean', 
                                                                     bins=tuple(np.linspace(-1,1,201)))
            
            fold_LC.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='0.5', lw=5,
                           label='mean')
            #fold_LC.plot(foldx1,foldy1,".",markersize=1,alpha=1,color="k",label="mean")
            fold_LC.plot(foldx1,foldy1,".",markersize=1,alpha=1,color="b",label="wk1_b")
            fold_LC.plot(foldx2,foldy2,".",markersize=1,alpha=0.5,color="r",label="wk2_r")
            fold_LC.plot(foldx3,foldy3,".",markersize=1,alpha=0.3,color="y",label="wk3_y")
            fold_LC.plot(foldx4,foldy4,".",markersize=1,alpha=0.3,color="c",label="wk4_c")
            leg = fold_LC.legend()


        
        data_hist   = fig.add_subplot(gs[3, 4])
        data_hist.set_title("Data Histogram",fontsize=10)
        raw_bins = np.linspace(np.min(self.TVOI.signal_cleaned),np.max(self.TVOI.signal_cleaned),101)
        data_hist.hist(self.TVOI.signal_cleaned,raw_bins,alpha=0.5,color="k",label="raw")
        signal_bin = self.TVOI.signal_bin_cleaned[self.TVOI.signal_bin_cleaned!=0]
        bin_bins = np.linspace(np.min(signal_bin),np.max(signal_bin),101)
        data_hist.hist(signal_bin,bin_bins,alpha=0.5,label="clean")
        data_hist.legend()
        data_hist.yaxis.set_visible(False)  
        
        
        
        
        gaia_plot   = fig.add_subplot(gs[3, 5])
        gaia_plot.set_title("Gaia Plot",fontsize=10)
        gaia.plot_gaia(ra,dec,gaia_data,gaia_plot)
        
        
        reserve = fig.add_subplot(gs[3, 6])
        
        
        reserve.set_title("Reserve",fontsize=10)
        
        if TVOI.num_flares > 0:
            reserve.set_title("Largest Flare",fontsize=10)
            
            
            max_flare_index = np.where(TVOI.signal_bin == max(TVOI.signal_bin[TVOI.flare_location]))[0][0]
            
            front, back = 75, 150
            
            reserve.plot(TVOI.time_bin[max_flare_index-front:max_flare_index+back]*TVOI.bin2day+TVOI.start_time, 
                         TVOI.signal_bin[max_flare_index-front:max_flare_index+back])
        
        
        
        
        #reserve_plot   = fig.add_subplot(gs[4, 0:])
        #reserve_plot.set_title("Gaia Plot",fontsize=10)
        #reserve_plot.plot_gaia(ra,dec,gaia_data,gaia_plot)
        
        
        if deploy:
            if int(self.num_peaks) == 1:
                plt.savefig("%s/L_TVOI_%s_%.2f_T%.1f_D%.2f_R%.2f_%s-%s-%s-%s-%s.png"%(self.savepath,TVOI.TIC_ID,
                                                        Tmag,Temp,distance,Rstar,
                                                        self.num_slow_peaks,self.num_low_peaks,self.num_mid_peaks,self.num_high_peaks,TVOI.num_flares))   
            
            else:
                plt.savefig("%s/H_TVOI_%s_%.2f_T%.1f_D%.2f_R%.2f_%s-%s-%s-%s-%s.png"%(self.savepath,TVOI.TIC_ID,
                                                        Tmag,Temp,distance,Rstar,
                                                        self.num_slow_peaks,self.num_low_peaks,self.num_mid_peaks,self.num_high_peaks,TVOI.num_flares))   
            plt.clf()
        else:
            plt.show()
        
        return 

    def Generate_ML_Plot(self,deploy=False,forced=False):
        
        if not forced and (len(self.sorted_peaks) == 0 or not self.should_generate):
            return
        
        TIC_ID = self.TVOI.TIC_ID
        
        fig = plt.figure(figsize=(6,6))
        FFT_LC = fig.add_axes([0,0,1,1])
        FFT_LC.loglog(self.freq,self.amplitude,"-",color="k",markersize=1)
        FFT_LC.set_xlim(0.5,100)
        FFT_LC.set_ylim(1e-8,1e-2)
        
        if deploy:
            plt.savefig("%s/FFT/FFT_%s.png"%(self.savepath,TIC_ID),dpi=50)
        else:
            plt.show()
        
        
    
        fig = plt.figure(figsize=(6,6))
        fold_LC = fig.add_axes([0,0,1,1])
        time_bin = self.TVOI.time_bin*self.TVOI.bin2day+self.TVOI.start_time
        signal_bin_bin = self.TVOI.signal_bin_cleaned.copy()
        
        a = np.array([time_bin,signal_bin_bin]).T
        new_time_bin,new_signal_bin = a[a[:,1]!=0].T
        
        foldx0,foldy0 = FFT.fold_lc(new_time_bin,new_signal_bin,self.predict_frequency,0)
        bin_means, bin_edges, binnumber = stats.binned_statistic(foldx0,foldy0, 
                                                                 statistic='mean', 
                                                                 bins=tuple(np.linspace(-1,1,201)))
        bin_centers = bin_edges[:-1]+(bin_edges[1:]-bin_edges[:-1])/2
        
        phase = bin_centers[np.argmin(bin_means)]
        
        foldx0,foldy0 = FFT.fold_lc(new_time_bin,new_signal_bin,self.predict_frequency,-phase)
        bin_means, bin_edges, binnumber = stats.binned_statistic(foldx0,foldy0, 
                                                                 statistic='mean', 
                                                                 bins=tuple(np.linspace(-1,1,201)))
        bin_centers = bin_edges[:-1]+(bin_edges[1:]-bin_edges[:-1])/2
        fold_LC.plot(bin_centers, bin_means, color="k")
        fold_LC.plot(foldx0,foldy0,".",markersize=2,alpha=1,color="k",label="mean")
        lower_threshold = np.percentile(foldy0,0.25)
        upper_threshold = np.percentile(foldy0,99.75)
        fold_LC.set_xlim(-1,1)
        fold_LC.set_ylim(lower_threshold*1.2,upper_threshold*1.2)
        if deploy:
            plt.savefig("%s/Data/Data_%s.png"%(self.savepath,TIC_ID),dpi=50)
        else:
            plt.show()
        
        
        
        fig = plt.figure(figsize=(6,6))
        Bin_LC = fig.add_axes([0,0,1,1])
        Bin_LC.plot(bin_centers, bin_means, color="k",linewidth=5, markersize=12)
        Bin_LC.set_xlim(-1,1)
        Bin_LC.set_ylim(lower_threshold*1.2,upper_threshold*1.2)
        if deploy:
            plt.savefig("%s/Bin/Bin_%s.png"%(self.savepath,TIC_ID),dpi=50)
        else:
            plt.show()
        
        return

class SPOC_Catcher_v3():
    """
    In the third version of the SPOC catcher, the thresholds are set by machine learning
    We will no longer break into multiple thresholds

    Initially we will save all the lightcurve txt files.
        An estimate should be made to figure out the size of output files
    """

    def __init__(self,filename,savepath,TIC_ID=0,sector=0,Norm=0,manual_cut=[]):
    
        self.filename = filename
        self.sector = sector
        self.Normalization = Norm
        self.savepath = savepath
        self.TIC_ID = TIC_ID
        self.manual_cut = manual_cut

        self.min = 0
        self.slow = 0
        self.low = 6
        self.mid = 48
        self.high = 96
        self.max = 360
        
        self.slow_thres = 11
        self.low_thres  = 11
        self.mid_thres  = 12
        self.high_thres = 12
        
        self.bump_frequency = True
        self.freq_catch_cut  = 1./5 # truncate frequency smaller than 1
        self.Outlier_Multiplier = 0.15

    def Load_Lightcurve_Data(self,cutoffs=[],user=False):
        
        self.cutoffs = cutoffs
        
        
        
        self.TVOI = LC_io2.SPOC_TVOI(self.filename,self.savepath,self.TIC_ID,self.sector,self.Normalization,self.manual_cut)
        
        
        if user != False:
            time,flux,error,tmag = user
            self.TVOI.load_user_input(time,flux,error,tmag)
            
            self.TVOI.calibrate_lightcurve(normalized=True,normed=1)
            
            self.TVOI.detrend_lightcurve()
            self.TVOI.bin_lightcurve()
        else:
            self.TVOI.load_object_data(cutoffs)

    def Conduct_FFT_Analysis_test(self):

        self.N1 = FFT.guess_N(len(self.TVOI.time_bin),8,False)
        self.extended_time, self.extended_signal = LCA.extend_lightcurve(self.TVOI.signal_bin_cleaned,self.N1,self.Normalization)
        
        
        FFT_Result = FFT.compute_fft_general(self.extended_signal, self.TVOI.time_step, self.N1)  
        self.freq,self.sig,self.amplitude,self.power = FFT_Result
        self.indexes = peakutils.indexes(self.amplitude,thres=0.5)#,min_dist=20)
        
        N = self.N1
        from scipy import fftpack
        
        time_step = self.TVOI.time_step
        y = self.extended_signal[:8192]
        x = self.extended_time[:8192]
        
        
        mean_y = np.mean(y)  # median vs mean... 1 point for mean 
        std_y = np.std(y)
        var_y = std_y**2
        print(mean_y,std_y,var_y)
        
        
        fft_signal = fftpack.fft(y)
        
        N = 8192
        fft_freq   = fftpack.fftfreq(N, d=time_step)           
        fft_theo_sci = 2.0*np.abs(fft_signal/N)
        
        mask = fft_freq >= 0
        
        ps = 2.0*(np.abs(fft_signal/N)**2)
        ps_sum = np.sum(ps[mask])
        print(ps_sum)
        
        
        #peaks,_ = find_peaks(fft_theo_sci[mask])

        
        fft_freq_real = fft_freq[mask]
        fft_theo_real = fft_theo_sci[mask]
        
        #print(peaks)
        power = np.abs(fft_signal)
        pos_mask = np.where(fft_freq > 0)
        peak_freq = fft_freq[power[pos_mask].argmax()]
        fft_new = fft_signal.copy()
        fft_new[np.abs(fft_freq) > peak_freq+0.5] = 0
        #fft_new[np.abs(fft_freq) < peak_freq-0.5] = 0
    
        
        
        filt_data = np.real(fftpack.ifft(fft_new))
        
        fft_new_theo = 2.0*np.abs(fftpack.fft(filt_data)/N)
        
        
        plt.subplot(3,1,1)
        plt.plot(x[:27*24*30],y[:27*24*30])
        #plt.plot(x[:27*24*30],filt_data[:27*24*30])
        plt.plot(x,filt_data)
        #plt.plot(x,filt_data)
        plt.subplot(3,1,2)
        #plt.plot(nwaves[mask],pow_var[mask])
        #plt.plot(freqs[mask],ps[mask])
        
        #plt.plot(freqs[mask],fft_new_theo[mask])
        plt.plot(fft_freq[mask],fft_theo_sci[mask])
        #plt.plot(fft_freq[peak],fft_theo_sci[peak],".")
        plt.plot(fft_freq[mask],fft_new_theo[mask])
        
        plt.subplot(3,1,3)
        plt.plot(x[:27*24*30],y[:27*24*30]-filt_data[:27*24*30])
        
        
        plt.show()

        
        
        #sys,exit()
        
        #plt.plot(fft_freq,fft_theo_sci2)
        #plt.show()
        
        # check if peak is harmonic, if so bypass significance
        
        
        # check if peak is significant

        #self.indexes = indexes
        
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
            
            if rel_new_peak_index == []:
                continue
            
            new_peak_index = rel_new_peak_index[0]+potential_peak_index-check_param
            
            """
            plt.plot(freq1[potential_peak_index-check_param:potential_peak_index+check_param],
                     amplitude1[potential_peak_index-check_param:potential_peak_index+check_param])
            plt.plot(freq1[new_peak_index],
                     amplitude1[new_peak_index],"o")
            plt.show()
            """
            

            
            frequency = freq1[new_peak_index]
            peak_frequencies.append(frequency)
            new_peak_indexs.append(new_peak_index)
        
        
        
        pfi = np.array(new_peak_indexs)
        self.period_flag = False
        
        if len(pfi) > 1 and self.bump_frequency:
            try:  # magic that I can't understand... presumably check if harmonic is present?
                pfi_index = [i for i, x in enumerate(pfi[1:]-int(pfi[0]/2) < 0.01*pfi[0]) if x][0]+1
                if pfi_index != 0:
                    self.predict_frequency = peak_frequencies[0]/2.
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

    def Generate_1Pager(self,deploy=False,forced=False):
        
        if not forced and (len(self.sorted_peaks) == 0 or not self.should_generate):
            return
        #print(self.should_generate)
        
        
        TVOI = self.TVOI
        
        try:
            Tmag  = float(TVOI.object["TESSMAG"])
        except:
            Tmag  = 0
        
        try:
            Rstar = float(TVOI.object["RADIUS"])
        except:
            Rstar = 0
            
        try:
            Temp  = float(TVOI.object["TEFF"])
        except:
            Temp = 0
            
        self.TVOI.Temp = Temp
        #Logg  = float(TVOI.object["LOGG"])
        #mH    = float(TVOI.object["MH"])
        try:
            ra    = float(TVOI.object["RA_OBJ"])
        except:
            ra = 0
        try:
            dec   = float(TVOI.object["DEC_OBJ"])
        except:
            dec = 0
        try:
            pmra  = float(TVOI.object["PMRA"])
        except:
            pmra = 0
        try:
            pmdec = float(TVOI.object["PMDEC"])
        except:
            pmdec = 0
        try:
            pmtot = float(TVOI.object["PMTOTAL"])
        except:
            pmtot = 0
        
        try:
            target_gaia_data = gaia.gaia_query(ra,dec,0.005)
        except:
            target_gaia_data = gaia.gaia_query(ra,dec,0.015)
            
        Gmag = min(target_gaia_data["Gmag"])
        parallex = max(target_gaia_data["RPlx"])/1000 #miliarcsec - arcsec
        distance = 1./parallex
        
        #print(distance)
        if distance == np.nan or str(distance) == "nan" or str(distance) == "--":
            distance = 0
        if Rstar == np.nan or str(Rstar) == "nan":
            Rstar = 0
            
            
        
        gaia_data = gaia.gaia_query(ra,dec,0.016)
        
        #if self.classification == None: 
        #    classification = "/"
        #else:
        
        #print(self.classification)
        try:
            classification = "/".join(self.classification)
        except:
            classification = "N"
        
        fig = plt.figure(figsize=(18,9),constrained_layout=True)#figsize=(18,9))#constrained_layout=True)
        gs = GridSpec(4,7, figure=fig)



        main_LC = fig.add_subplot(gs[0, 1:])
        main_LC.set_title("""TIC ID: %s (s:%s c:%s CCD:%s)                 T$_{mag}$: %.3f G$_{mag}$: %.3f T$_{eff}$: %sK R$_S$: %sR$\odot$ D$_S$: %.2fpc
                           \nFFT peaks: %s Flares: %s Classification:%s        Ra: %.5f Dec: %.5f pmra: %.4f pmdec: %.4f pmtot: %.4f"""%(
                        TVOI.TIC_ID,TVOI.sector,TVOI.camera,TVOI.ccd,
                        Tmag,Gmag,Temp,Rstar,distance,
                        self.num_peaks,TVOI.num_flares,classification,
                        ra,dec,pmra,pmdec,pmtot))
    
        
        
        main_LC.plot(TVOI.time_raw_uncut,TVOI.signal_calibrated_uncut,".",color="0.3",markersize=2,label="calibrated")
        main_LC.plot(TVOI.time_bin*TVOI.bin2day+TVOI.start_time,TVOI.signal_bin_cleaned,label="binned")
        main_LC.plot(TVOI.time_bin*TVOI.bin2day+TVOI.start_time,TVOI.filter_bin,label="trend")
        
        main_Max = np.max(TVOI.signal_bin_cleaned)*1.2
        main_Min = np.min(TVOI.signal_bin_cleaned)*1.2
        main_LC.set_ylim(main_Min,main_Max)
        
        
        
        zoomed_LC = fig.add_subplot(gs[0, 0])
        zoomed_LC.set_title("Zoomed LC (~2d)",fontsize=10)
        zoomed_low = 2048*2
        zoomed_high = 2048*2+1024
        zoomed_LC.plot(TVOI.time_calibrated[zoomed_low:zoomed_high]+TVOI.start_time,TVOI.signal_calibrated[zoomed_low:zoomed_high],".",markersize="2",color="0.3")
            
        
        

        
        
        raw_LC = fig.add_subplot(gs[1, 0])
        raw_LC.set_title("Raw Lightcurve",fontsize=10)
        raw_LC.plot(TVOI.time_raw_uncut,TVOI.signal_raw_uncut,".",markersize=2,color="0.3")
        raw_LC.xaxis.set_visible(False)
        raw_Max = np.max(TVOI.signal_raw)*1.05
        raw_Min = np.min(TVOI.signal_raw)*0.95
        raw_LC.set_ylim(raw_Min,raw_Max)
        
        centx_LC = fig.add_subplot(gs[2, 0])
        centx_LC.set_title("Centroid X/Y",fontsize=10)
        centx_LC.plot(TVOI.time_raw,TVOI.centdx-np.mean(TVOI.centdx),".",markersize=2,label="X")
        centx_LC.plot(TVOI.time_raw,TVOI.centdy-np.mean(TVOI.centdy),".",markersize=2,label="Y")
        centx_LC.xaxis.set_visible(False)
        centx_LC.legend()
        
        sonogram = fig.add_subplot(gs[3, 0])
        sonogram.set_title("Sonogram",fontsize=10)
        
        f, t, Sxx = scipy_signal.spectrogram(TVOI.signal_bin_cleaned,720)
        sonogram.pcolormesh(t, f, np.sqrt(Sxx))
        sonogram.set_ylim(0,48)
        #sonogram.set_ylabel('Frequency [Hz]')
        #sonogram.set_xlabel('Time [sec]')
        
        zoom_FFT    = fig.add_subplot(gs[1, 1:5])
        zoom_FFT.set_title("FFT ( 0 - 96 cycle/day)",fontsize=10)
        zoom_FFT.plot(self.freq,self.amplitude)
        zoom_FFT.plot(self.freq[self.peak_index],self.amplitude[self.peak_index],"*",color="b")
        try:
            zoom_FFT.plot(self.freq[self.sorted_peaks["index"]],self.amplitude[self.sorted_peaks["index"]],"*",color="r")
        except:
            pass
        zoom_FFT.set_xlim(0,100)


        
        log_FFT     = fig.add_subplot(gs[1, 5:7])
        log_FFT.set_title("Log FFT",fontsize=10)
        log_FFT.plot(self.freq,self.amplitude,label="Amp")
        log_FFT.plot(self.freq[self.slow_start_bin:self.slow_end_bin],self.slow_mean,color="k",label="Mean")
        log_FFT.plot(self.freq[self.slow_start_bin:self.slow_end_bin],self.slow_thres,color="r",label="Thres")
        log_FFT.plot(self.freq[self.low_start_bin:self.low_end_bin],self.low_mean,color="k")
        log_FFT.plot(self.freq[self.low_start_bin:self.low_end_bin],self.low_thres,color="r")
        log_FFT.plot(self.freq[self.mid_start_bin:self.mid_end_bin],self.mid_mean,color="k")
        log_FFT.plot(self.freq[self.mid_start_bin:self.mid_end_bin],self.mid_thres,color="r")   
        log_FFT.plot(self.freq[self.high_start_bin:self.high_end_bin],self.high_mean,color="k")
        log_FFT.plot(self.freq[self.high_start_bin:self.high_end_bin],self.high_thres,color="r")
        try:
            log_FFT.plot(self.freq[self.peak_index],self.amplitude[self.peak_index],"*",color="b")
            log_FFT.plot(self.freq[self.sorted_peaks["index"]],self.amplitude[self.sorted_peaks["index"]],"*",color="r")
        except:
            pass
        log_FFT.set_xlim(1,360)
        log_FFT.set_yscale("log")
        log_FFT.set_xscale("log")   
        log_FFT.legend()     
        
        slow_FFT    = fig.add_subplot(gs[2, 1])
        slow_FFT.set_title("FFT ( 0 - 6 cycle/day)",fontsize=10)
        slow_FFT.plot(self.freq,self.amplitude)
        slow_FFT.plot(self.freq[self.slow_start_bin:self.slow_end_bin],self.slow_mean,color="k")
        slow_FFT.plot(self.freq[self.slow_start_bin:self.slow_end_bin],self.slow_thres,color="r")
        slow_max = max(max(self.amplitude[(self.freq<=6)]),max(self.slow_thres))*1.2
        slow_FFT.set_xlim(0,6)     
        slow_FFT.set_ylim(0,slow_max)   
        slow_FFT.yaxis.set_visible(False) 
        
        low_FFT    = fig.add_subplot(gs[2, 2])
        low_FFT.set_title("FFT ( 6 - 48 cycle/day)",fontsize=10)
        low_FFT.plot(self.freq,self.amplitude)
        low_FFT.plot(self.freq[self.low_start_bin:self.low_end_bin],self.low_mean,color="k")
        low_FFT.plot(self.freq[self.low_start_bin:self.low_end_bin],self.low_thres,color="r")
        low_max = max(max(self.amplitude[(self.freq>=6) &(self.freq<=48)]),max(self.low_thres))*1.2
        low_FFT.set_xlim(6,48)      
        low_FFT.set_ylim(0,low_max)    
        low_FFT.yaxis.set_visible(False) 
        
        mid_FFT     = fig.add_subplot(gs[2, 3])
        mid_FFT.set_title("FFT ( 48 - 96 cycle/day)",fontsize=10)
        mid_FFT.plot(self.freq,self.amplitude)
        mid_FFT.plot(self.freq[self.mid_start_bin:self.mid_end_bin],self.mid_mean,color="k")
        mid_FFT.plot(self.freq[self.mid_start_bin:self.mid_end_bin],self.mid_thres,color="r") 
        mid_max = max(max(self.amplitude[(self.freq>=48) &(self.freq<=96)]),max(self.mid_thres))*1.2
        mid_FFT.set_xlim(48,96)  
        mid_FFT.set_ylim(0,mid_max)         
        mid_FFT.yaxis.set_visible(False) 
 
        high_FFT     = fig.add_subplot(gs[2, 4])
        high_FFT.set_title("FFT ( 96 - 360 cycle/day)",fontsize=10)
        high_FFT.plot(self.freq,self.amplitude)
        high_FFT.plot(self.freq[self.high_start_bin:self.high_end_bin],self.high_mean,color="k")
        high_FFT.plot(self.freq[self.high_start_bin:self.high_end_bin],self.high_thres,color="r")
        high_max = max(max(self.amplitude[(self.freq>=96)&(self.freq<=360)]),max(self.high_thres))*1.2
        high_FFT.set_xlim(96,360)  
        high_FFT.set_ylim(0,high_max)   
        high_FFT.yaxis.set_visible(False) 
        
        summed_FFT  = fig.add_subplot(gs[2, 5:7])
        summed_FFT.set_title("Summed FFT",fontsize=10)
        FFT_Summed = FFT.new_summed_fft(self.amplitude)
        summed_FFT.plot(self.freq,FFT_Summed)      
          

        fold_LC     = fig.add_subplot(gs[3, 1:4])
        if len(self.sorted_peaks)>=1:
            if self.period_flag:
                fold_LC.set_title("Detrended LC Folded @ P = %.6f days (%.3f hr)"%(1/self.predict_frequency,24/self.predict_frequency),fontsize=10,color="r")
            else:
                fold_LC.set_title("Detrended LC Folded @ P = %.6f days (%.3f hr)"%(1/self.predict_frequency,24/self.predict_frequency),fontsize=10)
        else:
            fold_LC.set_title("Detrended LC Not Folded",fontsize=10)
                    
            
             
        dp = int(len(self.TVOI.time_raw)/4)
        time_bin = self.TVOI.time_bin*self.TVOI.bin2day+self.TVOI.start_time
        signal_bin = self.TVOI.signal_bin_cleaned.copy()
        signal_bin_bin = self.TVOI.signal_bin_cleaned.copy()
        signal_bin[signal_bin==0] = np.nan
        
        if len(self.sorted_peaks) >= 1:
            foldx0,foldy0 = FFT.fold_lc(time_bin,signal_bin_bin,self.predict_frequency)
            foldx1,foldy1 = FFT.fold_lc(time_bin[:dp],signal_bin[:dp],self.predict_frequency)
            foldx2,foldy2 = FFT.fold_lc(time_bin[dp:2*dp],signal_bin[dp:2*dp],self.predict_frequency)
            foldx3,foldy3 = FFT.fold_lc(time_bin[2*dp:3*dp],signal_bin[2*dp:3*dp],self.predict_frequency)
            foldx4,foldy4 = FFT.fold_lc(time_bin[3*dp:],signal_bin[3*dp:],self.predict_frequency)        
            
            bin_means, bin_edges, binnumber = stats.binned_statistic(foldx0,foldy0, 
                                                                     statistic='mean', 
                                                                     bins=tuple(np.linspace(-1,1,201)))
            
            self.fold_lc_centers = (bin_edges[:-1] + bin_edges[1:])/2+bin_edges[:-1]
            self.fold_lc_bin_means = bin_means
            
            
            fold_LC.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='0.5', lw=5,
                           label='mean')
            #fold_LC.plot(foldx1,foldy1,".",markersize=1,alpha=1,color="k",label="mean")
            fold_LC.plot(foldx1,foldy1,".",markersize=1,alpha=1,color="b",label="wk1_b")
            fold_LC.plot(foldx2,foldy2,".",markersize=1,alpha=0.5,color="r",label="wk2_r")
            fold_LC.plot(foldx3,foldy3,".",markersize=1,alpha=0.3,color="y",label="wk3_y")
            fold_LC.plot(foldx4,foldy4,".",markersize=1,alpha=0.3,color="c",label="wk4_c")
            leg = fold_LC.legend()


        
        data_hist   = fig.add_subplot(gs[3, 4])
        data_hist.set_title("Data Histogram",fontsize=10)
        raw_bins = np.linspace(np.min(self.TVOI.signal_cleaned),np.max(self.TVOI.signal_cleaned),101)
        data_hist.hist(self.TVOI.signal_cleaned,raw_bins,alpha=0.5,color="k",label="raw")
        signal_bin = self.TVOI.signal_bin_cleaned[self.TVOI.signal_bin_cleaned!=0]
        bin_bins = np.linspace(np.min(signal_bin),np.max(signal_bin),101)
        data_hist.hist(signal_bin,bin_bins,alpha=0.5,label="clean")
        data_hist.legend()
        data_hist.yaxis.set_visible(False)  
        
        
    
        gaia_plot   = fig.add_subplot(gs[3, 5])
        gaia_plot.set_title("Nearby TIC",fontsize=10)
        
        nearby_star_info = get_exofop_nearby(self.TIC_ID)
        gaia_plot.plot(list(nearby_star_info["RA"]),list(nearby_star_info["Dec"]),"*",color="k")
        gaia_plot.plot(list(nearby_star_info["RA"])[0],list(nearby_star_info["Dec"])[0],"*",color="r")
        gaia_plot.plot(list(nearby_star_info["RA"])[0]+0.016,list(nearby_star_info["Dec"])[0]+0.016,".",color="y")
        gaia_plot.plot(list(nearby_star_info["RA"])[0]-0.016,list(nearby_star_info["Dec"])[0]-0.016,".",color="y")
        
        
        
        reserve = fig.add_subplot(gs[3, 6])
        
        
        reserve.set_title("Reserve",fontsize=10)
        
        if TVOI.num_flares > 0:
            reserve.set_title("Largest Flare",fontsize=10)
            
            
            max_flare_index = np.where(TVOI.signal_bin == max(TVOI.signal_bin[TVOI.flare_location]))[0][0]
            
            front, back = 75, 150
            
            reserve.plot(TVOI.time_bin[max_flare_index-front:max_flare_index+back]*TVOI.bin2day+TVOI.start_time, 
                         TVOI.signal_bin[max_flare_index-front:max_flare_index+back])
        
        
        
        
        #reserve_plot   = fig.add_subplot(gs[4, 0:])
        #reserve_plot.set_title("Gaia Plot",fontsize=10)
        #reserve_plot.plot_gaia(ra,dec,gaia_data,gaia_plot)
        
        try:
            Period = 1/self.predict_frequency
        except:
            Period = "NaN"
        if deploy:
            if int(self.num_peaks) == 1:
                filename_header = "L"
            elif int(self.num_peaks) <= 4:
                filename_header = "M"
            else:
                filename_header = "H"
            
            plt.savefig("%s/%s_TVOI_%s_%.2f_T%.1f_D%.2f_R%.2f_P%.5f_%s-%s-%s-%s-%s.png"%(self.savepath,filename_header,TVOI.TIC_ID,
                                                        Tmag,Temp,distance,Rstar,Period,
                                                        self.num_slow_peaks,self.num_low_peaks,self.num_mid_peaks,self.num_high_peaks,TVOI.num_flares))   
            plt.clf()
        else:
            plt.show()
        
        return 

    def Generate_ML_Plot(self,deploy=False,forced=False):
        
        if not forced and (len(self.sorted_peaks) == 0 or not self.should_generate):
            return
        
        TIC_ID = self.TVOI.TIC_ID
        
        fig = plt.figure(figsize=(6,6))
        FFT_LC = fig.add_axes([0,0,1,1])
        FFT_LC.loglog(self.freq,self.amplitude,"-",color="k",markersize=1)
        FFT_LC.set_xlim(0.5,100)
        FFT_LC.set_ylim(1e-8,1e-2)
        
        if deploy:
            plt.savefig("%s/FFT/FFT_%s.png"%(self.savepath,TIC_ID),dpi=50)
        else:
            plt.show()
        
        
    
        fig = plt.figure(figsize=(6,6))
        fold_LC = fig.add_axes([0,0,1,1])
        time_bin = self.TVOI.time_bin*self.TVOI.bin2day+self.TVOI.start_time
        signal_bin_bin = self.TVOI.signal_bin_cleaned.copy()
        
        a = np.array([time_bin,signal_bin_bin]).T
        new_time_bin,new_signal_bin = a[a[:,1]!=0].T
        
        foldx0,foldy0 = FFT.fold_lc(new_time_bin,new_signal_bin,self.predict_frequency,0)
        bin_means, bin_edges, binnumber = stats.binned_statistic(foldx0,foldy0, 
                                                                 statistic='mean', 
                                                                 bins=tuple(np.linspace(-1,1,201)))
        bin_centers = bin_edges[:-1]+(bin_edges[1:]-bin_edges[:-1])/2
        
        phase = bin_centers[np.argmin(bin_means)]
        
        foldx0,foldy0 = FFT.fold_lc(new_time_bin,new_signal_bin,self.predict_frequency,-phase)
        bin_means, bin_edges, binnumber = stats.binned_statistic(foldx0,foldy0, 
                                                                 statistic='mean', 
                                                                 bins=tuple(np.linspace(-1,1,201)))
        bin_centers = bin_edges[:-1]+(bin_edges[1:]-bin_edges[:-1])/2
        fold_LC.plot(bin_centers, bin_means, color="k")
        fold_LC.plot(foldx0,foldy0,".",markersize=2,alpha=1,color="k",label="mean")
        lower_threshold = np.percentile(foldy0,0.25)
        upper_threshold = np.percentile(foldy0,99.75)
        fold_LC.set_xlim(-1,1)
        fold_LC.set_ylim(lower_threshold*1.2,upper_threshold*1.2)
        if deploy:
            plt.savefig("%s/Data/Data_%s.png"%(self.savepath,TIC_ID),dpi=50)
        else:
            plt.show()
        
        
        
        fig = plt.figure(figsize=(6,6))
        Bin_LC = fig.add_axes([0,0,1,1])
        Bin_LC.plot(bin_centers, bin_means, color="k",linewidth=5, markersize=12)
        Bin_LC.set_xlim(-1,1)
        Bin_LC.set_ylim(lower_threshold*1.2,upper_threshold*1.2)
        if deploy:
            plt.savefig("%s/Bin/Bin_%s.png"%(self.savepath,TIC_ID),dpi=50)
        else:
            plt.show()
        
        return

    def Create_Flare_Report(self,
                            savepath="output_flare_sec1_Test",
                            deploy=False,
                            forced = False,
                            inject = False,
                            inject_param = [1,1,1]):
    
        if inject:
            injected_time = inject_param[0]
            injected_ampli = inject_param[1]
            injected_fwhms = inject_param[2]
    
        TOI            = self.TVOI
        trimmed_peaks  = TOI.flare_location
        
        mean           = TOI.flare_median
        rms            = TOI.flare_rms
        peak_threshold = TOI.peak_threshold
        wing_threshold = TOI.wing_threshold
        peak_num       = len(trimmed_peaks)
    
    
        fig = plt.figure(figsize=(16,9))#,constrained_layout=True)
        gs = GridSpec(5, 5, figure=fig)
        
        ax1 = fig.add_subplot(gs[0, 0:5])
        #ax2 = fig.add_subplot(gs[0, 1])
        #ax3 = fig.add_subplot(gs[0, 2])
        #ax4 = fig.add_subplot(gs[0, 3:])
        
        ax5 = fig.add_subplot(gs[1, 0])
        ax6 = fig.add_subplot(gs[1, 1])
        ax7 = fig.add_subplot(gs[1, 2])
        ax8 = fig.add_subplot(gs[1, 3])
        ax9 = fig.add_subplot(gs[1, 4])
        
        
        ax10 = fig.add_subplot(gs[2, 0])
        ax11 = fig.add_subplot(gs[2, 1])
        ax12 = fig.add_subplot(gs[2, 2])
        ax13 = fig.add_subplot(gs[2, 3])   
        ax14 = fig.add_subplot(gs[2, 4])    
        
        ax15 = fig.add_subplot(gs[3, 0])
        ax16 = fig.add_subplot(gs[3, 1])
        ax17 = fig.add_subplot(gs[3, 2])
        ax18 = fig.add_subplot(gs[3, 3])   
        ax19 = fig.add_subplot(gs[3, 4])  
        
        ax20 = fig.add_subplot(gs[4, 0])
        ax21 = fig.add_subplot(gs[4, 1])
        ax22 = fig.add_subplot(gs[4, 2])
        ax23 = fig.add_subplot(gs[4, 3])   
        ax24 = fig.add_subplot(gs[4, 4])  
    
    
        try:
            Tmag  = float(TOI.object["TESSMAG"])
        except:
            Tmag  = 0
        
        try:
            Rstar = float(TOI.object["RADIUS"])
        except:
            Rstar = 0
            
        try:
            Temp  = float(TOI.object["TEFF"])
        except:
            Temp = 0
        #Logg  = float(TOI.object["LOGG"])
        #mH    = float(TOI.object["MH"])
        try:
            ra    = float(TOI.object["RA_OBJ"])
        except:
            ra = 0
        try:
            dec   = float(TOI.object["DEC_OBJ"])
        except:
            dec = 0
        try:
            pmra  = float(TOI.object["PMRA"])
        except:
            pmra = 0
        try:
            pmdec = float(TOI.object["PMDEC"])
        except:
            pmdec = 0
        try:
            pmtot = float(TOI.object["PMTOTAL"])
        except:
            pmtot = 0
        
        if inject: 
            ax1.set_title("""TICID: %s \t Tmag:%.2f\tT$_{S}$:%sK \t R$_S$:%s \t s:%s c:%s CCD:%s peaks:%s/%s/%s i_time: %.4f i_amplitude: %.4f, i_fwhm: %.4f"""%(TOI.TIC_ID,Tmag,Temp,Rstar,TOI.sector,TOI.camera,TOI.ccd,
                 TOI.num_flares_medium,TOI.num_flares_global,TOI.num_flares,
                 inject_param[0],inject_param[1],inject_param[2]))
        else:
            ax1.set_title("""TICID: %s \t Tmag:%.2f\tT$_{S}$:%sK \t R$_S$:%s \t s:%s c:%s CCD:%s
            \n ra: %s dec: %s pmra: %s pmdec: %s pmtot: %s peaks:%s/%s/%s 
            \n injected time: %s injected amplitude: %s, injected fwhm: %s
            """%(TOI.TIC_ID,Tmag,Temp,Rstar,TOI.sector,TOI.camera,TOI.ccd,
                 ra,dec,pmra,pmdec,pmtot,
                 TOI.num_flares_medium,TOI.num_flares_global,TOI.num_flares,
                 inject_param[0],inject_param[1],inject_param[2]))
        
        
        
        #plt.plot(self.TVOI.time_bin*self.TVOI.bin2day,self.TVOI.signal_bin)
        ax1.plot(self.TVOI.time_raw,self.TVOI.signal_detrended)
        ax1.plot(self.TVOI.time_raw,self.TVOI.peak_threshold)
        ax1.plot(self.TVOI.time_raw,self.TVOI.wing_threshold)
        
        ax1.plot(self.TVOI.time_raw,self.TVOI.peak_threshold_medium,"--")
        ax1.plot(self.TVOI.time_raw,self.TVOI.wing_threshold_medium,"--")
        
        ax1.plot(self.TVOI.time_raw[self.TVOI.flare_location],
                 self.TVOI.signal_detrended[self.TVOI.flare_location],"*")
        
        ax1.axvline(injected_time,color="orange")
        
        
        # rasterized=True
        ax5.plot(self.TVOI.time_raw,self.TVOI.signal_detrended)
        
        
        ax10.plot(self.TVOI.time_raw,self.TVOI.centdx)
    
        ax15.plot(self.TVOI.time_raw,self.TVOI.centdy)   
        
        multiplier = (self.TVOI.signal_detrended[trimmed_peaks] - mean[trimmed_peaks])/rms[trimmed_peaks]
        peaks = list(zip(multiplier,trimmed_peaks))
        dtype = [('peak_rms', float), ('index', int)]
        sorted_peaks = np.sort(np.array(peaks, dtype=dtype), order='peak_rms')[::-1]    
        
        
        front, back = 75, 150
        plotter = [ax6,ax7,ax8,ax9,
                   ax11,ax12,ax13,ax14,
                   ax16,ax17,ax18,ax19,
                   ax21,ax22,ax23,ax24]
        
        datas = []

        for index, indiv_peaks in enumerate(sorted_peaks):
            
            peak  = indiv_peaks["index"]
            peak_rms = indiv_peaks["peak_rms"]
            abs = TOI.signal_detrended[peak] - mean[peak]
    
            duration = 1
            while True:
                try:
                    if TOI.signal_detrended[peak+duration] > TOI.signal_detrended[peak+duration+1]:
                        duration +=1
                    else:
                        break
                except:
                    break

            #print(index)
            if duration < 3:
                continue
            
            line = [index+1,TOI.time_raw[peak],"%.3f"%peak_rms,"%.3f"%abs,duration]
            datas.append(line)
            try:
                ax = plotter[index]
                ax.plot(TOI.time_raw[peak-front:peak+back], TOI.signal_detrended[peak-front:peak+back])
                ax.plot(TOI.time_raw[peak-front:peak+back], mean[peak-front:peak+back],alpha=0.3)
                ax.plot(TOI.time_raw[peak-front:peak+back], peak_threshold[peak-front:peak+back],alpha=0.3)
                ax.plot(TOI.time_raw[peak-front:peak+back], wing_threshold[peak-front:peak+back],alpha=0.3) 
                
                ax.plot(TOI.time_raw[peak-front:peak+back], TOI.peak_threshold_medium[peak-front:peak+back],"--")
                ax.plot(TOI.time_raw[peak-front:peak+back], TOI.wing_threshold_medium[peak-front:peak+back],"--") 
                
                  
                ax.plot(TOI.time_raw[peak:peak+duration+1], TOI.signal_detrended[peak:peak+duration+1],"*")  
                #ax.plot(TOI.time_bin[peak+1], TOI.signal_bin_filtered[peak+1],"*")         
                
                #ax.set_title("R:%.2f,A:%.2f,L:%.3f,D:%s"%(peak_rms,abs,TOI.time_raw[peak],duration))
                
                
                
                try:
                    ax5.plot( TOI.time_raw[peak-front:peak+back+duration+1],TOI.signal_detrended[peak-front:peak+back+duration+1],color="y")
                except:
                    pass
                
                try:
                    ax10.plot(TOI.time_raw[peak-front:peak+back+duration+1],TOI.centdx_bin[peak-front:peak+back+duration+1],color="y")
                except:
                    pass
                
                try:
                    ax15.plot(TOI.time_raw[peak-front:peak+back+duration+1],TOI.centdy_bin[peak-front:peak+back+duration+1],color="y")
                except:
                    pass
                
                
            except:
                break
    
    
    
        ax5.set_ylabel("Flux")
        ax10.set_ylabel("Centdx")
        ax15.set_ylabel("Centdy")
        
        
        
        if deploy:
            plt.savefig("%s/Flare_Plot_P%s_S%s_%s.png"%(savepath,peak_num,TOI.sector,TOI.TIC_ID))
            plt.clf()
            #print("saved")
        else:
            plt.show()

        
        with open("%s/Flare_Info_P%s_S%s_%s.csv"%(savepath,peak_num,TOI.sector,TOI.TIC_ID), mode='w') as file:
            flare_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            flare_writer.writerow(["Flare Number","TJD","RMS","Flux Increase(%)","Duration"])
            
            for infos in datas:
                flare_writer.writerow(infos)

        return peak_num
        
class SPOC_Catcher_Tester():
    
    def __init__(self,sector=3):
    
        self.sector = sector

    def test_numpy_savetxt(self):
    
        filepath = "../../input/a.Example_SPOC_LC/Sector3_General"
        filenames = glob.glob("%s/*.gz"%filepath)
        
        #print(filenames)
    
        np.savetxt("filename.txt",filenames,fmt="%s")
        
        filenames = np.loadtxt("filename.txt",dtype="str")
        #print(filenames)

    def test_SPOC_TVOI_loader(self):
        
        filepath = "../../input/a.Example_SPOC_LC/Sector3_General"
        
        savepath = ""
        sector = self.sector
        Norm = 0
        filenames = glob.glob("%s/*.gz"%filepath)
        
        
        for filename in filenames:

            print(filepath)
            
            TIC_ID = str(int(filename.split("/")[-1].split("-")[2]))
            TVOI = LC_io.SPOC_TVOI(TIC_ID,sector,Norm,filename,None)
            TVOI.load_object_data(721)
            
            
            
            
        
            print(TVOI.object["TESSMAG"])
            print(TVOI.flare_location)
            
            """
            plt.plot(TVOI.time_calibrated,TVOI.signal_raw,".")
            plt.plot(TVOI.time_calibrated*TVOI.day2bin,TVOI.signal_calibrated,".")
            plt.show()
            """
            
            plt.plot(TVOI.time_bin,TVOI.signal_bin,".")
            plt.plot(TVOI.time_bin,TVOI.lc_filter)
            plt.plot(TVOI.time_bin,TVOI.signal_bin_detrended)
            plt.plot(TVOI.time_bin,TVOI.peak_threshold)
            plt.plot(TVOI.time_bin,TVOI.wing_threshold)
            plt.show() 
            
            plt.plot(TVOI.time_bin, TVOI.signal_bin_detrended)
            plt.plot(TVOI.time_bin, TVOI.signal_bin_cleaned)
            plt.show() 

    def test_FFT_Calculation(self):

        filepath = "../../input/a.Example_SPOC_LC/Sector3_General"
        filepath = "../../input/a.Example_SPOC_LC/Sector1_Interesting_Mdwarf"
        savepath = ""
        sector = self.sector
        sector = 1
        
        filenames = glob.glob("%s/*.gz"%filepath)
        
        # normalization factor
        Norm = 0
        
        
        for filename in filenames:
            start_catch_time = time.time()
            
            TIC_ID = str(int(filename.split("/")[-1].split("-")[2]))
            if TIC_ID != "206544316":
                continue
            
            Catcher = SPOC_Catcher(filename,savepath,sector,Norm)
            Catcher.Load_Lightcurve_Data()
            Catcher.Conduct_FFT_Analysis()
            Catcher.Generate_1Pager(deploy=False) 


            """
            plt.plot(self.freq,self.amplitude)
            plt.plot(self.freq[slow_start_bin:slow_end_bin],self.amplitude[slow_start_bin:slow_end_bin])
            plt.plot(self.freq[low_start_bin:low_end_bin]  ,self.amplitude[low_start_bin:low_end_bin])
            plt.plot(self.freq[mid_start_bin:mid_end_bin]  ,self.amplitude[mid_start_bin:mid_end_bin])
            plt.plot(self.freq[high_start_bin:high_end_bin],self.amplitude[high_start_bin:high_end_bin])
            
            plt.plot(self.freq[slow_start_bin:slow_end_bin],slow_mean,color="k")
            plt.plot(self.freq[slow_start_bin:slow_end_bin],slow_thres,color="r")
            
            plt.plot(self.freq[low_start_bin:low_end_bin],low_mean,color="k")
            plt.plot(self.freq[low_start_bin:low_end_bin],low_thres,color="r")
            
            plt.plot(self.freq[mid_start_bin:mid_end_bin],mid_mean,color="k")
            plt.plot(self.freq[mid_start_bin:mid_end_bin],mid_thres,color="r")   
                 
            plt.plot(self.freq[high_start_bin:high_end_bin],high_mean,color="k")
            plt.plot(self.freq[high_start_bin:high_end_bin],high_thres,color="r")
            
            plt.plot(self.freq[peak_index],self.amplitude[peak_index],"*")
            plt.plot(self.freq[sorted_peaks["index"]],self.amplitude[sorted_peaks["index"]],"*")
                         
            #plt.yscale("log")
            plt.xlim(0,12)
            plt.show()
            """
            
            """
            plt.plot(Catcher.extended_time,Catcher.extended_signal)
            plt.show()
            """
            
    def test_Generate_1Pager(self):

        filepath = "../../input/a.Example_SPOC_LC/Sector3_General"
        #filepath = "../../input/a.Example_SPOC_LC/Sector1_Interesting_Mdwarf"
        savepath = "."
        sector = self.sector
        sector = 1
        Norm = 0 # normalization factor
        filenames = glob.glob("%s/*.gz"%filepath)
    
        for filename in filenames:
            
            filename = "../../input/TOI/1/tess2018206045859-s0001-0000000140068425-0120-s_lc.fits.gz"
            
            start_catch_time = time.time()
            
            Catcher = SPOC_Catcher(filename,savepath,sector,Norm,121)
            Catcher.Load_Lightcurve_Data()
            Catcher.Conduct_FFT_Analysis()
            Catcher.Generate_1Pager(deploy=False) 
            
class QLP_Catcher():
    
    def __init__(self,filename,savepath,sector=0,cam=0,ccd=0,
                 Norm=0,detrending=49,low_thres=13,high_thres=13):
    
        self.filename = filename
        
        self.sector = sector
        self.camera = cam
        self.ccd = ccd
        
        self.Normalization = Norm
        self.savepath = savepath
        self.detrending = detrending
        
        self.timestep = 1/48
        self.N1 = 8
        self.N2 = 128
        self.local_pt = 2**8
        
        # here are regions that "just works"
        # eventually I want to optimize this for physical objects
        self.min = 0
        self.slow = 0
        self.low = 12
        self.max = 24
        
        self.slow_thres = low_thres
        self.low_thres  = high_thres
        
        self.bump_frequency = True
        self.freq_catch_cut  = 1 # truncate frequency smaller than 1
        self.freq_catch_cut2  = 23.8 # truncate frequency higher than 1
        self.Outlier_Multiplier = 0.15       
        
    def Load_Lightcurve_Data(self):

        self.TVOI = LC_io.QLP_TVOI(self.filename,self.savepath,self.sector,self.Normalization)
        self.TVOI.load_object_data(self.detrending)   
        
        self.TIC_ID = self.TVOI.TIC_ID     

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
        
        self.freq,self.amplitude = PA.fft_transform(self.TVOI.time_bin,
                                                    self.TVOI.signal_bin_cleaned,
                                                    oversampling=self.N1,timestep=self.timestep)
        
        
        
        self.total_bin_num  = len(self.freq)
        self.slow_start_bin = int(self.total_bin_num*self.slow/self.max)
        self.low_start_bin  = int(self.total_bin_num*self.low/self.max)
        
        self.slow_end_bin   = pad_array_index(self.slow_start_bin,self.low_start_bin,self.local_pt)
        self.low_end_bin    = pad_array_index(self.low_start_bin,self.total_bin_num,self.local_pt)

        self.slow_mean, self.slow_thres = self.Detection_Threshold(self.amplitude[self.slow_start_bin:self.slow_end_bin],
                                                         self.slow_thres,self.local_pt*2,"Boxcar Mean",offset=self.slow_start_bin,prevmean=True)        
        self.low_mean, self.low_thres   = self.Detection_Threshold(self.amplitude[self.low_start_bin:self.low_end_bin],
                                                         self.low_thres,self.local_pt, "Boxcar Mean",offset=self.low_start_bin)
        
        self.slow_peaks,_ = find_peaks(self.amplitude[self.slow_start_bin:self.slow_end_bin],self.slow_thres)
        self.low_peaks,_ = find_peaks(self.amplitude[self.low_start_bin:self.low_end_bin],self.low_thres)

        
        self.num_slow_peaks = len(self.slow_peaks)
        self.num_low_peaks = len(self.low_peaks)
        
        # half the threshold
        self.peak_index_lower,_ = find_peaks(self.amplitude[self.slow_start_bin:self.slow_end_bin],self.slow_thres/2)
        self.peak_index_side,_ = find_peaks(self.amplitude[self.slow_start_bin:self.slow_end_bin],self.slow_thres*0.8)
        
        
        self.peak_index = list(set(list(self.slow_peaks+self.slow_start_bin)+
                             list(self.low_peaks+self.low_start_bin)))
        
        self.num_peaks = len(self.peak_index)
        self.side_peaks = len(self.peak_index_side)
        self.lower_peaks = len(self.peak_index_lower)
        
        
        """
        plt.plot(self.freq,self.amplitude)
        plt.plot(self.freq[self.slow_start_bin:self.slow_end_bin],self.slow_mean)
        plt.plot(self.freq[self.slow_start_bin:self.slow_end_bin],self.slow_thres)
        plt.plot(self.freq[self.low_start_bin:self.low_end_bin],self.low_mean)
        plt.plot(self.freq[self.low_start_bin:self.low_end_bin],self.low_thres)
        plt.plot(self.freq[self.peak_index],self.amplitude[self.peak_index],".")
        plt.show()
        """


        if self.num_peaks < 1:
            self.period_flag = False
            self.should_generate = False
            self.sorted_peaks = []
            self.predict_frequency = 0
            return 0
            
        peaks_lower = list(zip(self.amplitude[self.peak_index_lower],self.peak_index_lower))
        peaks = list(zip(self.amplitude[self.peak_index],self.peak_index))
        dtype = [('power', float), ('index', int)]
        self.sorted_peaks = np.sort(np.array(peaks, dtype=dtype), order='power')[::-1][:3]
        
        freq1,amplitude1 = PA.fft_transform(self.TVOI.time_bin,
                                            self.TVOI.signal_bin_cleaned,
                                            oversampling=128,timestep=1/48)

        check_param = int(self.N2/self.N1)*2 # get the amplitude of the function 
        incident_peak_index = self.sorted_peaks[0]["index"]
        potential_peak_index = int(incident_peak_index*self.N2/self.N1)
        
        
        if potential_peak_index-check_param < 0:
            front = 0
        else:
            front = potential_peak_index-check_param 
        rel_new_peak_index,_ = find_peaks(amplitude1[front:potential_peak_index+check_param],
                                amplitude1[potential_peak_index-check_param])
        
        new_peak_index = rel_new_peak_index[0]+potential_peak_index-check_param
        self.predict_frequency = freq1[new_peak_index]
        
        self.bump_frequency = False
        for peak in peaks_lower:
            if peak[1] > incident_peak_index:
                continue
            if (abs(int(incident_peak_index/2) - peak[1]) < int(incident_peak_index/2)*0.01):
                self.predict_frequency = self.predict_frequency/2
                self.bump_frequency = True
            
        # truncate any period longer than 1 days or shorter than 1.002 hr.

        if self.predict_frequency < self.freq_catch_cut and self.side_peaks < 4:
            self.should_generate = False
            return 0
        elif self.predict_frequency > self.freq_catch_cut2:
            self.should_generate = False
            return 0
        else:
            self.should_generate = True      
            return 1             

    def Generate_1Pager(self,deploy=False,forced=False):
        
        if not forced and (len(self.sorted_peaks) == 0 or not self.should_generate):
            return
        #print(self.should_generate)
        
        
        TVOI = self.TVOI
        
        
        star_info = get_star_info(TVOI.TIC_ID)
        
        StarName = star_info["Name"]
        Tmag  = star_info["Tmag"]
        Rstar = star_info["Radius"]
        Mstar = star_info["Mass"]
        Temp  = star_info["Temperature"]
        ra    = star_info["Ra"]
        dec   = star_info["Dec"]
        pmra  = star_info["Pmra"]
        pmdec = star_info["Pmdec"]
        Rmag  = star_info["Rmag"]
        distance = star_info["Distance"]
        
        """
        if self.should_generate:
            classification = "/"
        else:
            classification = "Bad"
        """
        self.classification = ""
        print(self.classification)
        
        classification = "/".join(self.classification)
        
        fig = plt.figure(figsize=(18,9))#,constrained_layout=True)#figsize=(18,9))#constrained_layout=True)
        gs = GridSpec(4,7, figure=fig, wspace=0.5, hspace=0.5)
        
        

        main_LC = fig.add_subplot(gs[0, 1:7])
        
        #main_LC.set_title("TIC ID: %s"%TVOI.TIC)
        
        main_LC.set_title("""TIC ID:%s (S:%s C:%s CCD:%s)              Ra: %.5f Dec: %.5f pmra: %.4f pmdec: %.4f         
                      FFT peaks: %s/%s/%s Classification:%s            T$_{mag}$: %s R$_{mag}$: %s T$_{eff}$: %sK R$_S$: %sR$\odot$ D$_S$: %.2fpc"""%(
                        TVOI.TIC_ID,self.sector,self.camera,self.ccd,
                        ra,dec,pmra,pmdec,
                        self.num_peaks,self.side_peaks,self.lower_peaks,classification,
                        Tmag,Rmag,Temp,Rstar,distance), fontsize=12)
        
        
        
        main_LC.plot(TVOI.time_raw,TVOI.signal_calibrated,".",color="0.3",markersize=2,label="calibrated")
        main_LC.plot(TVOI.time_bin*TVOI.bin2day+TVOI.start_time,TVOI.signal_bin_cleaned,label="binned")
        main_Max = np.max(TVOI.signal_bin_cleaned)*1.2
        main_Min = np.min(TVOI.signal_bin_cleaned)*1.2
        main_LC.set_ylim(main_Min,main_Max)
        
        
        
        zoomed_LC = fig.add_subplot(gs[0, 0])
        zoomed_LC.set_title("Zoomed LC (~1d)",fontsize=10)
        zoomed_low = 144
        zoomed_high = 144+48
        zoomed_LC.plot(TVOI.time_calibrated[zoomed_low:zoomed_high]+TVOI.start_time,TVOI.signal_calibrated[zoomed_low:zoomed_high],"o-",markersize="2",color="0.3")
            
        
        
        raw_LC = fig.add_subplot(gs[1, 0])
        raw_LC.set_title("Raw Lightcurve",fontsize=10)
        raw_LC.plot(TVOI.time_raw,TVOI.signal_raw,".",markersize=2,color="0.3")
        raw_LC.xaxis.set_visible(False)
        raw_Max = np.max(TVOI.signal_raw)*1.05
        raw_Min = np.min(TVOI.signal_raw)*0.95
        raw_LC.set_ylim(raw_Min,raw_Max)
        
        centx_LC = fig.add_subplot(gs[2, 0])
        centx_LC.set_title("Centroid X/Y",fontsize=10)
        centx_LC.plot(TVOI.time_raw,TVOI.centdx-np.mean(TVOI.centdx),".",markersize=2,label="X")
        centx_LC.plot(TVOI.time_raw,TVOI.centdy-np.mean(TVOI.centdy),".",markersize=2,label="Y")
        centx_LC.xaxis.set_visible(False)
        centx_LC.legend()
        
        sonogram = fig.add_subplot(gs[2, 6])
        sonogram.set_title("Sonogram",fontsize=10)
        
        f, t, Sxx = scipy_signal.spectrogram(TVOI.signal_bin_cleaned,48)
        sonogram.pcolormesh(t, f, np.sqrt(Sxx))
        sonogram.set_ylim(0,24)
        #sonogram.set_ylabel('Frequency [Hz]')
        #sonogram.set_xlabel('Time [sec]')
        
        zoom_FFT    = fig.add_subplot(gs[1, 1:4])
        zoom_FFT.set_title("FFT ( 0 - 24 cycle/day)",fontsize=10)
        zoom_FFT.plot(self.freq,self.amplitude)
        zoom_FFT.plot(self.freq[self.peak_index],self.amplitude[self.peak_index],"*",color="b")
        zoom_FFT.plot(self.freq[self.slow_start_bin:self.slow_end_bin-2**7],self.slow_mean[:-2**7],color="0.2")
        zoom_FFT.plot(self.freq[self.slow_start_bin:self.slow_end_bin-2**7],self.slow_thres[:-2**7],color="r")
        zoom_FFT.plot(self.freq[self.low_start_bin+2**7:self.low_end_bin],self.low_mean[2**7:],color="0.4")
        zoom_FFT.plot(self.freq[self.low_start_bin+2**7:self.low_end_bin],self.low_thres[2**7:],color="g")

        try:
            zoom_FFT.plot(self.freq[self.sorted_peaks["index"]],self.amplitude[self.sorted_peaks["index"]],"*",color="r")
        except:
            pass
        zoom_FFT.set_xlim(0,25)    
        
        
        summed_FFT  = fig.add_subplot(gs[1, 4:7])
        summed_FFT.set_title("Summed FFT",fontsize=10)
        FFT_Summed = FFT.new_summed_fft(self.amplitude)
        summed_FFT.plot(self.freq,FFT_Summed)      
          
        FFT_ML = fig.add_subplot(gs[3, 0])
        FFT_ML.set_title("FFT ML")
        FFT_ML.loglog(self.freq,self.amplitude,".",color="k",markersize=1)
        FFT_ML.set_xlim(0.5,25)



        fold_LC     = fig.add_subplot(gs[2, 1:4])
        if len(self.sorted_peaks)>=1:
            if self.bump_frequency:
                fold_LC.set_title("Detrended LC Folded @ P = %.6f days (%.3f hr)"%(1/self.predict_frequency,24/self.predict_frequency),fontsize=10,color="r")
            else:
                fold_LC.set_title("Detrended LC Folded @ P = %.6f days (%.3f hr)"%(1/self.predict_frequency,24/self.predict_frequency),fontsize=10)
        else:
            fold_LC.set_title("Detrended LC Not Folded",fontsize=10)
                    
            
             
        dp = int(len(self.TVOI.time_bin)/4)
        time_bin = self.TVOI.time_bin*self.TVOI.bin2day+self.TVOI.start_time
        signal_bin = self.TVOI.signal_bin_cleaned.copy()
        signal_bin_bin = self.TVOI.signal_bin_cleaned.copy()
        signal_bin[signal_bin==0] = np.nan
        
        if len(self.sorted_peaks) >= 1:
            
            foldxx,foldyy = FFT.fold_lc(TVOI.time_calibrated,TVOI.signal_cleaned,self.predict_frequency)
            foldx0,foldy0 = FFT.fold_lc(time_bin,signal_bin_bin,self.predict_frequency)
            foldx1,foldy1 = FFT.fold_lc(time_bin[:dp],signal_bin[:dp],self.predict_frequency)
            foldx2,foldy2 = FFT.fold_lc(time_bin[dp:2*dp],signal_bin[dp:2*dp],self.predict_frequency)
            foldx3,foldy3 = FFT.fold_lc(time_bin[2*dp:3*dp],signal_bin[2*dp:3*dp],self.predict_frequency)
            foldx4,foldy4 = FFT.fold_lc(time_bin[3*dp:],signal_bin[3*dp:],self.predict_frequency)        
            
            lower_threshold = np.percentile(foldy0,1)
            upper_threshold = np.percentile(foldy0,99)
            
            
            bin_means, bin_edges, binnumber = stats.binned_statistic(foldx0,foldy0, 
                                                                     statistic='mean', 
                                                                     bins=tuple(np.linspace(-1,1,101)))
            bin_centers = bin_edges[:-1]+(bin_edges[1:]-bin_edges[:-1])/2
            
            #fold_LC.plot(foldx1,foldy1,".",markersize=1,alpha=1,color="k",label="mean")
            fold_LC.plot(foldx1,foldy1,".",markersize=5,alpha=1,color="b",label="wk1_b")
            fold_LC.plot(foldx2,foldy2,".",markersize=5,alpha=1,color="r",label="wk2_r")
            fold_LC.plot(foldx3,foldy3,".",markersize=5,alpha=1,color="y",label="wk3_y")
            fold_LC.plot(foldx4,foldy4,".",markersize=5,alpha=1,color="c",label="wk4_c")
            fold_LC.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors="k", lw=5,
                           label='mean')
            fold_LC.set_ylim(lower_threshold*1.2,upper_threshold*1.2)
            leg = fold_LC.legend()


        fold_LC2     = fig.add_subplot(gs[3, 1])
        if len(self.sorted_peaks)>=1:
            fold_LC2.set_title("ML LC Bin Bin",fontsize=10)
            fold_LC2.plot(bin_centers,bin_means, color="k")
            fold_LC2.hlines(bin_means, bin_edges[:-1]-0.01, bin_edges[1:]+0.01, colors="k", lw=5,
                                   label='mean')
            fold_LC2.set_ylim(lower_threshold*1.2,upper_threshold*1.2)
            
        else:
            pass

        if len(self.sorted_peaks) >= 1:

            foldx0nan = np.array(foldx0).copy()
            foldy0nan = np.array(foldy0).copy()
            
            
            
            
            foldx0nan[foldy0nan==0]=np.nan
            foldy0nan[foldy0nan==0]=np.nan
            
            
            fold_LC3     = fig.add_subplot(gs[3, 2])
            if len(self.sorted_peaks)>=1:
                fold_LC3.set_title("ML LC Bin",fontsize=10)
                fold_LC3.plot(foldx0nan,foldy0nan,".",markersize=2,alpha=0.3,color="k",label="mean")
                fold_LC3.set_ylim(lower_threshold*1.2,upper_threshold*1.2)
            else:
                pass
    
            fold_LC4     = fig.add_subplot(gs[3, 3])
            if len(self.sorted_peaks)>=1:
                fold_LC4.set_title("ML LC Unbin",fontsize=10)
                fold_LC4.plot(foldxx,foldyy,".",markersize=2,alpha=0.3,color="k",label="mean")
                fold_LC4.set_ylim(lower_threshold*1.2,upper_threshold*1.2)
            else:
                pass

        
        data_hist   = fig.add_subplot(gs[2, 4])
        data_hist.set_title("Data Histogram",fontsize=10)
        raw_bins = np.linspace(np.min(self.TVOI.signal_cleaned),np.max(self.TVOI.signal_cleaned),101)
        data_hist.hist(self.TVOI.signal_cleaned,raw_bins,alpha=0.5,color="k",label="raw")
        signal_bin = self.TVOI.signal_bin_cleaned[self.TVOI.signal_bin_cleaned!=0]
        bin_bins = np.linspace(np.min(signal_bin),np.max(signal_bin),101)
        data_hist.hist(signal_bin,bin_bins,alpha=0.5,label="clean")
        data_hist.legend()
        data_hist.yaxis.set_visible(False)  
        
        
        gaia_plot   = fig.add_subplot(gs[2, 5])
        gaia_plot.set_title("Nearby TIC",fontsize=10)
        
        nearby_star_info = get_exofop_nearby(self.TIC_ID)
        gaia_plot.plot(list(nearby_star_info["RA"]),list(nearby_star_info["Dec"]),"*",color="k")
        gaia_plot.plot(list(nearby_star_info["RA"])[0],list(nearby_star_info["Dec"])[0],"*",color="r")
        gaia_plot.plot(list(nearby_star_info["RA"])[0]+0.016,list(nearby_star_info["Dec"])[0]+0.016,".",color="y")
        gaia_plot.plot(list(nearby_star_info["RA"])[0]-0.016,list(nearby_star_info["Dec"])[0]-0.016,".",color="y")
        
        
        text_plot   = fig.add_subplot(gs[3, 4:7])
        text_plot.set_title("Additional Information")
        #text_plot.axis('off')
        text_plot.set_yticklabels([])
        text_plot.set_xticklabels([])
        
        newstarname = []
        for count,i in enumerate(StarName.split(",")):
            newstarname.append(i)
            if count%2 == 0:
                newstarname.append("\n")

        text_plot.text(0.5, 0.5, '%s'%"".join(newstarname),
                          ha='center', va='center', size=12, wrap=True)
        
        

        
        
        
        
        
        if deploy:
            
            
            
            if (self.side_peaks == 1 and self.predict_frequency < self.freq_catch_cut) or self.should_generate == 0: 
                plt.savefig("%s/L_%s%s%s_%s_%.5f.png"%(self.savepath,self.sector,self.camera,self.ccd,
                                                 TVOI.TIC,self.predict_frequency),
                                                       bbox_inches = 'tight',pad_inches = 0.2)
            else:
                plt.savefig("%s/H_%s%s%s_%s_%.5f.png"%(self.savepath,self.sector,self.camera,self.ccd,
                                                     TVOI.TIC,self.predict_frequency),
                                                       bbox_inches = 'tight',pad_inches = 0.2)
            
            """
            if int(self.num_peaks) == 1:
                plt.savefig("%s/L_TVOI_%s_%.2f_T%.1f_D%.2f_R%.2f_%s-%s-%s-%s-%s.png"%(self.savepath,TVOI.TIC_ID,
                                                        Tmag,Temp,distance,Rstar,
                                                        self.num_slow_peaks,self.num_low_peaks,self.num_mid_peaks,self.num_high_peaks,TVOI.num_flares))   
            
            else:
                plt.savefig("%s/H_TVOI_%s_%.2f_T%.1f_D%.2f_R%.2f_%s-%s-%s-%s-%s.png"%(self.savepath,TVOI.TIC_ID,
                                                        Tmag,Temp,distance,Rstar,
                                                        self.num_slow_peaks,self.num_low_peaks,self.num_mid_peaks,self.num_high_peaks,TVOI.num_flares))   
            """
            #plt.cla()
            plt.clf()
            #plt.close()
        else:
            plt.show()
        
        return 
        
        



if __name__ == "__main__":
    
    
    """
    sector = 3
    Catcher_tester = SPOC_Catcher_Tester(sector)
    #Catcher.test_SPOC_TVOI_loader()
    #Catcher_tester.test_FFT_Calculation()
    Catcher_tester.test_Generate_1Pager()
    #Catcher_tester.test_numpy_savetxt()
    """
    """
    filename = "test_data/206544316.h5"
    savepath = ""
    sector =  2
    Norm  = 0
    
    Catcher = QLP_Catcher(filename,savepath,sector,Norm)
    Catcher.Load_Lightcurve_Data()
    output = Catcher.Conduct_FFT_Analysis()

    """
    
    sector = 9
    name = "Test"
    norm = 0
    
    filepaths = glob.glob("../../input/Sector9/problematic/*.gz")
    
    savepath = "output_sec%s_%s"%(sector,name)
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    
        
    for count,filename in enumerate(filepaths):
        print(filename)
        
        #if "48063032" not in filename:
        #    continue
        TIC_ID = int(filename.split("/")[-1].split("-")[2])

        Catcher = SPOC_Catcher_v3(filename,savepath,TIC_ID,sector,norm)
        Catcher.Load_Lightcurve_Data()
        output = Catcher.Conduct_FFT_Analysis()
        
        TVOI = Catcher.TVOI
        
        plt.subplot(2, 1, 1)
        #plt.plot(TVOI.time_raw,TVOI.signal_raw)
        plt.plot(TVOI.time_bin,TVOI.signal_bin_cleaned)
        plt.plot(TVOI.time_bin,TVOI.signal_bin_detrended)

        
        plt.subplot(2, 1, 2)
        plt.plot(Catcher.freq,Catcher.amplitude)
        plt.plot(Catcher.freq[Catcher.slow_start_bin:Catcher.slow_end_bin], Catcher.slow_thres)
        plt.plot(Catcher.freq[Catcher.low_start_bin:Catcher.low_end_bin], Catcher.low_thres)
        plt.plot(Catcher.freq[Catcher.mid_start_bin:Catcher.mid_end_bin], Catcher.mid_thres)
        plt.plot(Catcher.freq[Catcher.high_start_bin:Catcher.high_end_bin], Catcher.high_thres)
        #plt.plot(Catcher.freq[Catcher.indexes],Catcher.amplitude[Catcher.indexes],"*")
        plt.xscale("log")
        plt.show()
        
    









