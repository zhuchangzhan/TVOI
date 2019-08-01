"""

This is the lightcurve loader version 4

"""


import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d,LSQUnivariateSpline
from astropy.io import fits
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel

DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(DIR, '../..'))


class TVOI():
    """
    master class to inherit from.
    self.ip[""] = 
    
    """
    
    def __init__(self,ip=[]):

        if ip != []:
            self.ip = ip
        else:
            self.ip = "test"
        
        self.ip["TJD_Offset"] = 2457000.0
    
    
    def load_data(self):
        pass




class SPOC_TVOI():
    
    def __init__(self,ip):
        """
        
        """
        TVOI.__init__(self,ip)
        
        self.ip["cadence"] = 2.
        self.ip["time_step"] = 1/720.
        self.ip["day2bin"] = 24*30.
        self.ip["bin2day"] = 1/(24*30.)
        
        
        for i in self.ip:
            print(i,self.ip[i])
        


class QLP_TVOI():
    
    def __init__(self,ip):
        TVOI.__init__(self,ip)

        self.ip["cadence"] = 30.
        self.ip["time_step"] = 1/48.
        self.ip["day2bin"] = 24*2.
        self.ip["bin2day"] = 1/(24*2.)
        
        self.mag2flux_0 = 1.48*10e7
        self.data_pts_per_sector = 1336
        self.std = 4 # flare and bad data remove    
    
if __name__ == "__main__":
    
    import glob
    sector = 9
    name = "Test"
    savepath = "../../output/TVOI_TEST/output_sec%s_%s"%(sector,name)
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    
    filepaths = glob.glob("../../input/Sector%s/*.gz"%sector)    
    for count,filename in enumerate(filepaths):
        #print(filename)
        TIC = int(filename.split("/")[-1].split("-")[2])
        
        input = {}
        input["filename"] = filename
        input["savepath"] = savepath
        input["sector"] = sector
        
        input["TIC_ID"] = TIC  
        input["TIC"] = str(TIC)
        
        input["normalization"] = 0
        input["manual_cut"] = []
        
        
        TVOI = SPOC_TVOI(input)
    
        break
    

    
    
    
    
    
    
    
    
    