# -*- coding: utf-8 -*-
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

import src.main.LC_Analysis as LCA
from src.util.star_query import get_exofop_info

class TVOI():
    """
    master class to inherit from.
    """
    
    def __init__(self,filename=None,savepath=None,TIC_ID=0,sector=0,Norm=1,manual_cut=None):

        self.filename = filename
        self.savepath = savepath 
        self.sector = sector
        self.Normalization = Norm
        self.manual_cut = manual_cut   
        
        self.TIC_ID = TIC_ID
        self.TIC = str(TIC_ID)
        self.TJD_Offset = 2457000.0
          
        self.std = 4

    def load_object_information_and_lightcurve(self):
        pass

    def calibrate_lightcurve(self,normalized=False,normed=0):

        if self.sector == 0:
            print("No Sector info inputed")
            sys.exit()
        elif self.sector == 1:
            crop_location = []#[9121,10136]]#[[1,2500]]
        elif self.sector == 2:
            crop_location = []#[[1,2500]]
        elif self.sector == 3:
            crop_location = [[0,2600]]
        elif self.sector == 4:
            crop_location = []#[[]]
        elif self.sector == 5:
            crop_location = []
        elif self.sector == 6:
            crop_location = []
        elif self.sector == 7:
            crop_location = []
        else:
            crop_location = []
        
 

        self.time_raw_uncut = self.time_raw.copy()
        self.signal_raw_uncut = self.signal_raw.copy()
        
        for a,b in crop_location:
            
            self.time_raw   = np.concatenate([self.time_raw[0:a],self.time_raw[b:]])
            self.signal_raw = np.concatenate([self.signal_raw[0:a],self.signal_raw[b:]])
            self.signal_error_raw = np.concatenate([self.signal_error_raw[0:a],self.signal_error_raw[b:]])
            self.centdx     = np.concatenate([self.centdx[0:a],self.centdx[b:]])
            self.centdx_err = np.concatenate([self.centdx_err[0:a],self.centdx_err[b:]])
            self.centdy     = np.concatenate([self.centdy[0:a],self.centdy[b:]])
            self.centdy_err = np.concatenate([self.centdy_err[0:a],self.centdy_err[b:]])
            
            

        self.time_calibrated = self.time_raw - self.start_time
        self.time_calibrated_uncut = self.time_raw_uncut - self.start_time

        lower_threshold = np.nanpercentile(self.signal_raw,5)
        upper_threshold = np.nanpercentile(self.signal_raw,95)
        
        truncated_signal = self.signal_raw.copy()
        truncated_signal = truncated_signal[(self.signal_raw > lower_threshold)&(self.signal_raw < upper_threshold)]


        if normalized:
            self.signal_calibrated       = self.signal_raw - normed
            self.signal_calibrated_uncut = self.signal_raw_uncut - normed
            self.signal_error_calibrated = self.signal_error_raw - normed
        else:
            self.signal_calibrated = self.signal_raw/np.median(truncated_signal) + self.Normalization -1
            self.signal_calibrated_uncut = self.signal_raw_uncut/np.median(truncated_signal) + self.Normalization -1
            self.signal_error_calibrated = self.signal_error_raw/np.median(truncated_signal)

    def detrend_lightcurve(self,data_gap=0.02,spline_day=7):
        """
        detrending now happen on the raw data.
        detrending also solved the data gap/orbit problem it seems
        Should have the ability to insert manual points for detrending
        
        data_gap is the minimum split time. 
        spline_day is number of day for the split
        """
        
        data_gap = 0.02
    
        
        self.signal_filter    = []
        self.signal_detrended = []
        self.signal_cleaned   = []
        self.signal_error_cleaned = []
        
        deltaT = self.time_calibrated[1:]-self.time_calibrated[:-1]        
        seperator = []
        for i in deltaT:
            if i > data_gap:  # if the gap is larger than half a hour. This need a second look
                seperator.append(np.where(deltaT == i)[0][0]+1)
        prev = 0
        seperator.append(-1)
        
        
        start = True
        for sep in seperator:
            
            if sep == -1:
                time = self.time_calibrated[prev:]
                signal = self.signal_calibrated[prev:]
                error = self.signal_error_calibrated[prev:]
            else:
                time = self.time_calibrated[prev:sep]
                signal = self.signal_calibrated[prev:sep]
                error = self.signal_error_calibrated[prev:sep]
            
            
            #plt.plot(time,signal)
            if len(time) == 1:
                
                filler = np.zeros(len(time))
                if self.signal_detrended != []:
                    
                    self.signal_filter        = np.concatenate([self.signal_filter,filler])
                    self.signal_detrended     = np.concatenate([self.signal_detrended,filler])
                    self.signal_cleaned       = np.concatenate([self.signal_cleaned,filler])
                    self.signal_error_cleaned = np.concatenate([self.signal_error_cleaned,filler])
                else:
                    self.signal_filter        = np.array(filler)
                    self.signal_detrended     = np.array(filler)
                    self.signal_cleaned       = np.array(filler)
                    self.signal_error_cleaned = np.array(filler)
                #print("error")
                if sep != -1:
                    prev = sep
                else:
                    prev = len(self.time_calibrated)
                
                #print("here")
                continue
            


            
            lower_threshold = np.nanpercentile(signal,5)
            upper_threshold = np.nanpercentile(signal,95)
            
            truncated = signal.copy()
            truncated[(signal > upper_threshold)] = upper_threshold
            truncated[(signal < lower_threshold)] = lower_threshold 
            truncated = np.nan_to_num(truncated)
            
            chunk = int((time[-2]-time[1])/spline_day)
           
            if chunk == 0:
                if sep == -1:
                    tarray = [time[int(len(time)/2)]]
                else:
                    tarray = [time[int((sep-prev)/2)]]
            else:
                tarray = np.linspace(time[0],time[-1],chunk)
            
            
            
            
            sp = LSQUnivariateSpline(np.concatenate([[time[0]-1],time,[time[-1]+1]]),
                                     np.concatenate([[truncated[0]],truncated,[truncated[-1]]]),
                                     t=tarray)
            lc_filter1 = sp(time)
            
            
            if sep == -1:
                signal_detrended = self.signal_calibrated[prev:]-lc_filter1
            else:
                signal_detrended = self.signal_calibrated[prev:sep]-lc_filter1
                
            rms = np.std(signal_detrended)
            signal_error   = error
            signal_cleaned = signal_detrended.copy()
            signal_cleaned[signal_cleaned > self.std*rms] = 0  
            # this removes flares in a way, but ....
            
            
            if start:
                start = False
                self.signal_filter        = np.array(lc_filter1)
                self.signal_detrended     = np.array(signal_detrended)
                self.signal_cleaned       = np.array(signal_cleaned)
                self.signal_error_cleaned = np.array(signal_error)
      
            else:
                self.signal_filter        = np.concatenate([self.signal_filter,lc_filter1])
                self.signal_detrended     = np.concatenate([self.signal_detrended,signal_detrended])
                self.signal_cleaned       = np.concatenate([self.signal_cleaned,signal_cleaned])
                self.signal_error_cleaned = np.concatenate([self.signal_error_cleaned,signal_error])

            
            """
            plt.plot(self.time_calibrated[prev:sep],signal_detrended)
            plt.plot(self.time_calibrated[prev:sep],signal_clean)
            
            # plotting the detrending
            plt.plot(self.time_calibrated[prev:sep],self.signal_calibrated[prev:sep],".")
            plt.plot(self.time_calibrated[prev:sep],self.signal_calibrated[prev:sep]-lc_filter1,"-")
            
            plt.show()
            """
            if sep != -1:
                prev = sep
            else:
                prev = len(self.time_calibrated)
        
        """
        plt.plot(self.time_raw_uncut-self.time_raw_uncut[0],self.signal_raw_uncut,".")
        plt.plot(self.time_calibrated,self.signal_calibrated,".")
        plt.plot(self.time_calibrated,self.signal_detrended,".")
        plt.plot(self.time_calibrated,self.signal_cleaned,".")
        
        
        plt.show()
        """
        
        #self.flare_location_small,self.flare_median_small,self.flare_rms_small,self.peak_threshold_small,self.wing_threshold_small = LCA.find_flares(self.signal_detrended,self.object["TESSMAG"],2**4)
        #self.num_flares_small = len(self.flare_location_small)
        
        self.flare_location_medium,self.flare_median_medium,self.flare_rms_medium,self.peak_threshold_medium,self.wing_threshold_medium = LCA.find_flares(self.signal_detrended,self.object["TESSMAG"],2**7,"boxcar_mean")
        self.num_flares_medium = len(self.flare_location_medium)
        
        
        self.flare_location_global,self.flare_median_global,self.flare_rms_global,self.peak_threshold_global,self.wing_threshold_global = LCA.find_flares(self.signal_detrended,self.object["TESSMAG"],2**10,"boxcar_mean")
        self.num_flares_global = len(self.flare_location_global)
        
        #self.flare_location = list(set(self.flare_location_small+self.flare_location_medium+self.flare_location_global))
        self.flare_location = list(set(self.flare_location_medium+self.flare_location_global))
        
        self.flare_median = self.flare_median_global
        self.flare_rms = self.flare_rms_global
        self.peak_threshold = self.peak_threshold_global
        self.wing_threshold = self.wing_threshold_global
        
        self.num_flares = len(self.flare_location)
        print(self.num_flares_medium,self.num_flares_global,self.num_flares)
        #print(self.flare_location,self.num_flares)
        
             
    def bin_lightcurve(self):

        self.data_pts = int((self.end_time-self.start_time)*self.day2bin) 
        self.data_pts+=1
        self.time_bin = np.arange(self.data_pts)


        time_floor = np.array(np.floor(self.time_calibrated*self.day2bin),dtype=int)
        self.signal_bin           = np.zeros(self.data_pts)
        self.signal_bin_detrended = np.zeros(self.data_pts)
        self.signal_bin_cleaned   = np.zeros(self.data_pts)
        self.signal_error_bin     = np.zeros(self.data_pts)
        self.filter_bin           = np.zeros(self.data_pts)
        self.centdx_bin           = np.zeros(self.data_pts)
        self.centdy_bin           = np.zeros(self.data_pts)
        
        
        self.signal_bin[time_floor] = self.signal_calibrated
        self.signal_bin_detrended[time_floor] = self.signal_detrended
        self.signal_bin_cleaned[time_floor] = self.signal_cleaned
        self.signal_error_bin[time_floor] = self.signal_error_cleaned
        self.filter_bin[time_floor] = self.signal_filter
        self.centdx_bin[time_floor] = self.centdx
        self.centdy_bin[time_floor] = self.centdy
        
        
        if self.sector == 0:
            print("No Sector info inputed")
            sys.exit()
        elif self.sector == 1:
            crop_location = [[9521,10343],[15650,17650]]
        elif self.sector == 2:
            crop_location = [[9390,10435]]
        elif self.sector == 3:
            crop_location = [[0,2900],[9650,10520]]
        elif self.sector == 4:
            crop_location = [[1680,1700],[5460,7450],[8500,9850],[18600,-1]]
        elif self.sector == 5:
            crop_location = [[0,20],[8600,9800],[18400,-1]]
        elif self.sector == 6:
            crop_location = [[6250,7150]]
        elif self.sector == 7:
            crop_location = [[8200,9440]]
        elif self.sector == 8:
            crop_location = [[0,500],[8400,14000]]
        elif self.sector == 9:
            crop_location = [[0,900],[8850,10520],[11040,11430]]
        else:
            crop_location = []
        
        
        for a,b in crop_location:
            if b == -1:
                end = len(self.signal_bin)
                self.signal_bin[a:] = np.ones(end-a)*self.Normalization
                self.signal_bin_detrended[a:] = np.ones(end-a)*self.Normalization
                self.signal_bin_cleaned[a:] = np.ones(end-a)*self.Normalization
                self.signal_error_bin[a:] = np.ones(end-a)*self.Normalization
            elif a == 0:
                self.signal_bin[:b] = np.ones(b)*self.Normalization
                self.signal_bin_detrended[:b] = np.ones(b)*self.Normalization
                self.signal_bin_cleaned[:b] = np.ones(b)*self.Normalization
                self.signal_error_bin[:b] = np.ones(b)*self.Normalization
                
            else:
                self.signal_bin[a:b] = np.ones(b-a)*self.Normalization
                self.signal_bin_detrended[a:b] = np.ones(b-a)*self.Normalization
                self.signal_bin_cleaned[a:b] = np.ones(b-a)*self.Normalization
                self.signal_error_bin[a:b] = np.ones(b-a)*self.Normalization
            
    def load_object_data(self,cutoff):
        
        self.cutoff = cutoff
        self.load_object_information_and_lightcurve()
        self.calibrate_lightcurve()
        self.detrend_lightcurve()
        self.bin_lightcurve()

class SPOC_TVOI(TVOI):
    
    def __init__(self,filename=None,savepath=None,TIC_ID=0,sector=0,Norm=1,manual_cut=None):
        TVOI.__init__(self,filename,savepath,TIC_ID,sector,Norm,manual_cut)
        
        self.cadence = 2.
        self.time_step = 1/720.
        self.day2bin = 24*30.
        self.bin2day = 1/self.day2bin
        
        self.pdc = False
    
    def load_user_input(self,time,flux,error,tmag):
        
        
        #self.sector = 0
        self.camera= 0
        self.ccd= 0
        
        self.object = {}
        if tmag != -1:
            self.object["TESSMAG"] = tmag

        else:
            info = get_exofop_info(self.TIC_ID) # only care about the last output
            Magnitude_Information = info[-1]
            TESSmag = Magnitude_Information["TESS"]
            
            try: 
                self.object["TESSMAG"] = float(TESSmag)
            except:
                self.object["TESSMAG"] = float(TESSmag.split("±")[0])


        
        self.time_raw = time - self.TJD_Offset
        self.time_raw_uncut = self.time_raw
        self.signal_raw = flux
        self.signal_raw_error = error
        self.signal_error_raw = error

        zeros = np.zeros(len(self.signal_raw))

        self.centdx     = zeros
        self.centdx_err = zeros
        self.centdy     = zeros
        self.centdy_err = zeros
        
        
        self.start_time = self.time_raw[0]
        self.end_time = self.time_raw[-1]
        self.start_time_uncut = self.time_raw_uncut[0]
        self.end_time_uncut = self.time_raw_uncut[0]
        
        if self.sector == 4:
            self.orbit_gap = self.time_raw[np.argsort(self.time_raw[1:]-self.time_raw[:-1])[-2]]
        else:
            self.orbit_gap = self.time_raw[np.argmax(self.time_raw[1:]-self.time_raw[:-1])]
        
    def load_object_information_and_lightcurve(self):
        
        # load data header
        hdul = fits.open(self.filename)
        self.hdul_header = hdul[0].header
        
        TIC_ID = int(hdul[0].header["TICID"])
        if TIC_ID != self.TIC_ID:
            print("TIC_ID mismatch, %s vs %s"%(TIC_ID,self.TIC_ID))
            
        self.sector = hdul[0].header["SECTOR"]
        self.camera = hdul[0].header["CAMERA"]
        self.ccd    = hdul[0].header["CCD"]
        
        self.Tstart = hdul[0].header["TSTART"]   # Note the slight difference between Tstart and actual data Time[0]
        self.Tstop  = hdul[0].header["TSTOP"]
        
        self.object = {}
        self.object["TESSMAG"] = hdul[0].header["TESSMAG"]
        self.object["TEFF"]    = hdul[0].header["TEFF"]
        #self.object["LOGG"]    = vars(hdul[0].header["LOGG"])
        self.object["LOGG"]    = 0.0
        self.object["RADIUS"]  = hdul[0].header["RADIUS"]
        #self.object["MH"]      = hdul[0].header["MH"]
        self.object["MH"]      = 0.0
        self.object["RA_OBJ"]  = hdul[0].header["RA_OBJ"]
        self.object["DEC_OBJ"] = hdul[0].header["DEC_OBJ"]
        self.object["PMRA"]    = hdul[0].header["PMRA"]
        self.object["PMDEC"]   = hdul[0].header["PMDEC"]
        self.object["PMTOTAL"] = hdul[0].header["PMTOTAL"]

        self.df = Table.read(self.filename).to_pandas()  
        
        self.pdc = True  
        
        if self.pdc:
            self.filter_df = self.df[(self.df.QUALITY == 0) & 
                           (self.df.PDCSAP_FLUX != np.nan)]# bad flag filter 
            self.notnull = self.filter_df.PDCSAP_FLUX.notnull()
            self.signal_raw = np.array(self.filter_df.PDCSAP_FLUX[self.notnull])
            self.signal_error_raw = np.array(self.filter_df.PDCSAP_FLUX_ERR[self.notnull])
            self.signal_raw_uncut = np.array(self.df.PDCSAP_FLUX)
        else:
            self.filter_df = self.df[(self.df.QUALITY == 0) & 
                           (self.df.SAP_FLUX != np.nan)]# bad flag filter
            self.notnull = self.filter_df.SAP_FLUX.notnull()  # the indexing method for df doesn't completely remove np.nan. 
            self.signal_raw = np.array(self.filter_df.SAP_FLUX[self.notnull])
            self.signal_error_raw = np.array(self.filter_df.SAP_FLUX_ERR[self.notnull])       
            self.signal_raw_uncut = np.array(self.df.SAP_FLUX)
            
        self.centdx     = self.filter_df.MOM_CENTR1[self.notnull]
        self.centdx_err = self.filter_df.MOM_CENTR1_ERR[self.notnull]
        self.centdy     = self.filter_df.MOM_CENTR2[self.notnull]
        self.centdy_err = self.filter_df.MOM_CENTR2_ERR[self.notnull]

        self.time_raw_uncut = np.array(self.df.TIME)
        self.time_raw = np.array(self.filter_df.TIME[self.notnull])
        
        
        self.start_time = self.time_raw[0]
        self.end_time = self.time_raw[-1]
        self.start_time_uncut = self.time_raw_uncut[0]
        self.end_time_uncut = self.time_raw_uncut[0]

        # seperate sector into orbits by finding the index of where the data gap is
        # this needs to happen before removing bad lightcurve location
        
        if self.sector == 4:
            self.orbit_gap = self.time_raw[np.argsort(self.time_raw[1:]-self.time_raw[:-1])[-2]]
        else:
            self.orbit_gap = self.time_raw[np.argmax(self.time_raw[1:]-self.time_raw[:-1])]
        
class QLP_TVOI(TVOI):
    
    def __init__(self,filename=None,savepath=None,TIC_ID=0,orbit=0,Norm=1,manual_cut=None):
        
        TVOI.__init__(self,filename,savepath,TIC_ID,orbit,Norm,manual_cut)
        
        self.cadence = 30.
        self.time_step = 1/48.
        self.day2bin = 24*2.
        self.bin2day = 1/self.day2bin
        
        self.mag2flux_0 = 1.48*10e7
        self.data_pts_per_sector = 1336
        self.std = 4 # flare and bad data remove    

    def load_object_information_and_lightcurve(self):

        file = h5py.File(self.filename, 'r')
        

        
        Magnitude = file["CatalogueMagnitudes"]
        Lightcurve = file["LightCurve"]
        
        AperturePhotometry  = Lightcurve["AperturePhotometry"]
        Background          = Lightcurve["Background"]
        BJD                 = Lightcurve["BJD"] 
        Cadence             = Lightcurve["Cadence"]
        Centroid_X          = Lightcurve["X"]
        Centroid_Y          = Lightcurve["Y"]
        
        # how do you use these? the values doesn't match with either flux or magnitude that well
        Background_Error = Background["Error"]
        Background_Value = Background["Value"]

        KSP_Magnitude = np.array(AperturePhotometry["Aperture_002"]["KSPMagnitude"]) 
        Target_Magnitude = np.array(AperturePhotometry["Aperture_002"]["RawMagnitude"])
        Target_Mag_Error = np.array(AperturePhotometry["Aperture_002"]["RawMagnitudeError"])



        # Convert magnitude to flux
        Target_Flux = 10**(-Target_Magnitude/2.5)*self.mag2flux_0
        
        """
        if self.sector > 1:
            QualityFlag = Lightcurve["QFLAG"]
        else:
        """
        
        QualityFlag = AperturePhotometry["Aperture_002"]["QualityFlag"]
        
        
        self.orig_df = pd.DataFrame({"Flag":QualityFlag,"time_raw":BJD,"signal_raw":Target_Flux,
                           "centdx":Centroid_X,"centdy":Centroid_Y,"magnitude":Target_Magnitude,
                           "mag_error":Target_Mag_Error,"ksp_mag":KSP_Magnitude})



        self.signal_orig = self.orig_df["signal_raw"]
        
        momentum_dump = []
        self.raw_df = self.orig_df.drop(momentum_dump)
        
        self.raw_df = self.raw_df[pd.notnull(self.raw_df["signal_raw"]) ]
        # will have to do for now.
        self.Magnitude = np.median(self.raw_df["magnitude"])
        self.object = {}
        self.object["TESSMAG"] = self.Magnitude
                
        
        self.time_raw = np.array(self.raw_df["time_raw"])
        self.signal_raw = np.array(self.raw_df["signal_raw"])
        self.signal_error_raw = np.array(self.raw_df["mag_error"])
        self.magnitudes = np.array(self.raw_df["magnitude"])
        self.ksp_mag = np.array(self.raw_df["ksp_mag"])
        self.centdx = np.array(self.raw_df["centdx"])
        self.centdy = np.array(self.raw_df["centdy"])
        
        self.time_raw_uncut = np.array(self.orig_df["time_raw"])
        self.signal_raw_uncut = np.array(self.orig_df["signal_raw"])
        self.magnitude_uncut  = np.array(self.orig_df["magnitude"])
        
        self.start_time  = np.array(self.raw_df["time_raw"])[0]
        self.end_time   = np.array(self.raw_df["time_raw"])[-1]

        self.start_time_uncut  = np.array(self.orig_df["time_raw"])[0]
        self.end_time_uncut   = np.array(self.orig_df["time_raw"])[-1]

if __name__ == "__main__":
    
    import glob
    
    
    sector = 9
    norm = 0
    name = "Test"
    savepath = "../../output/TVOI_TEST/output_sec%s_%s"%(sector,name)
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    
    filepaths = glob.glob("../../input/Sector%s/*.gz"%sector)    
    for count,filename in enumerate(filepaths):
        print(filename)
        TIC_ID = int(filename.split("/")[-1].split("-")[2])
        TVOI = SPOC_TVOI(filename,savepath,TIC_ID,sector,norm)
        TVOI.load_object_information_and_lightcurve()
        TVOI.calibrate_lightcurve()
        TVOI.detrend_lightcurve()
        TVOI.bin_lightcurve()
    
        plt.plot(TVOI.time_bin,TVOI.signal_bin_cleaned)
        plt.show()
        break
    """
    
    orbit = 24
    norm = 0
    name = "Test"
    savepath = "../../output/TVOI_TEST/output_orb%s_%s"%(orbit,name)
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    
    #filepaths = glob.glob("../../input/Sector%s/*.gz"%sector) 
    filepaths = glob.glob("../../input/orbit-%s/cam1/ccd1/*.h5"%orbit)    
    for count,filename in enumerate(filepaths):
        print(filename)
        TIC_ID = int(filename.split("/")[-1].replace(".h5",""))
        
        TVOI = QLP_TVOI(filename,savepath,TIC_ID,orbit,norm)
        TVOI.load_object_information_and_lightcurve()
    
        plt.plot(TVOI.time_raw,TVOI.signal_raw)
        plt.show()
        break
    
    """
    
    
    
    
    
    
    