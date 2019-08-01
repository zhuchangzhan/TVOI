"""

Catcher optimized for catching high amplitude variations

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

DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(DIR, '../..'))

import src.util.configurable as config
import src.main.Lightcurve_io3 as LC_io


class SPOC_High_Amp_Catcher():
    
    def __init__(self,filename,savepath,sector=0,Norm=0,detrending=721):
    
        self.filename = filename
        self.sector = sector
        self.Normalization = Norm
        self.savepath = savepath
        self.detrending = detrending
        
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
        self.high_thres = 11
        
        self.bump_frequency = True
        self.freq_catch_cut  = 1 # truncate frequency smaller than 1
        self.Outlier_Multiplier = 0.15

    def Load_Lightcurve_Data(self):

        self.TVOI = LC_io.SPOC_TVOI(self.filename,self.savepath,self.sector,self.Normalization)
        self.TVOI.load_object_data(self.detrending)  
        
    
    def Conduct_High_Amp_Analysis(self):
        
        raw_min = np.min(self.TVOI.signal_raw)
        raw_max = np.max(self.TVOI.signal_raw)
        
        max_min_ratio = raw_max/raw_min
        percentile_ratio = np.percentile(self.TVOI.signal_raw,95)/np.percentile(self.TVOI.signal_raw,5)
        
        
        
        
        
        
        
          