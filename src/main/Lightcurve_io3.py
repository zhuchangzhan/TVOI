"""

Rewrite Lightcurve_io2 with the explicit ability to properly process multi sector data

The question is how beneficial is it to process multi sector data for high frequency oscillations.

Is it really worth the trouble? What's the compelling reason? 
Perhaps because we want to find high frequency oscillations alongside EB and planet like objects? 
Mostly because of planet like objects...

But we're not creating a planet detector. The FFT is not good for that either. 

So.... Hold off on developing the multi-sector loader but keep it possible at least?

Planets will already be indentified by SPOC main pipeline so... We could just repeat using a FFT Analyzer.

Adding multisector is alot of work without obvious benefits. I might kick myself later but....

Therefore for now have an explicit single sector lightcurve processor?


strictly single sector processor, pretty much a mock up of LC_io2


        TVOI can be called from both a TIC ID or filepath.
        Right now I'm working with a filepath but in the future this will 
        more likely to be called from a TIC ID.
        
        cadence and num_bin is not really used in version one since I'm only
        dealing with SPOC data. I should write another wrapper to deal with qlp data.
        At present stage it's not wise to create a "do all" processor.
        
        TVOI: TIC ID, usually a int or string, both is fine
        filepath: Path to data
        savepath: Location to store output data.
        
        Removed Sector because it doesn't care



For Version 4:
    use df for SPOC as well
    generalize QLP and SPOC to take data from more sources
    Post to github

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
import src.main.FFT_Analysis as FFT

class TVOI():
    """
    Should think about implementing inheritance in the future
    """
    
    def __init__(self):
        """
        
        """
        pass


class QLP_TVOI():
    """
    Loading QLP Lightcurve data
    """
    
    def __init__(self,filename=None,savepath=None,sector=0,Norm=0):
        self.filename = filename
        self.savepath = savepath #Not used
        self.sector = sector
        self.Normalization = Norm
        
        self.TIC = filename.split("/")[-1].replace(".h5","")
        self.TIC_ID = self.TIC
        
        self.cadence = 30.
        self.time_step = 1/48.
        self.day2bin = 24*2.
        self.bin2day = 1/self.day2bin

        self.TJD_Offset = 2457000.0
        
        self.mag2flux_0 = 1.48*10e7
        
        self.data_pts_per_sector = 1336
        
        self.std = 4 # flare and bad data remove

    def load_object_data(self,detrending=49):
        
        self.load_object_information_and_lightcurve()
        self.remove_momentum_dump()
        self.calibrate_lightcurve()
        self.detrend_lightcurve(2)
        self.bin_lightcurve()
        #self.crop_bin_lightcurve("1")
        
        
        """
        with open("%s_raw.txt"%self.TIC_ID, "w") as f:
            for x,y in zip(self.time_raw,self.signal_raw):
                f.write("%s %s\n"%(x,y))
        """
        
        #self.detrend_bin_lightcurve(detrending)
        #self.remove_momentum_dump2()
        #self.crop_bin_lightcurve()
        
        """
        plt.plot(self.time_bin,self.signal_bin)
        plt.plot(self.time_bin,self.signal_cleaned)
        plt.plot(self.time_bin,self.signal_bin_cleaned)
        """
        #plt.plot(self.signal_raw)
        #plt.show()

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
        
        
        if self.sector > 1:
            QualityFlag = Lightcurve["QFLAG"]
        else:
            QualityFlag = AperturePhotometry["Aperture_002"]["QualityFlag"]
        
        
        self.orig_df = pd.DataFrame({"Flag":QualityFlag,"time_raw":BJD,"signal_raw":Target_Flux,
                           "centdx":Centroid_X,"centdy":Centroid_Y,"magnitude":Target_Magnitude,
                           "mag_error":Target_Mag_Error,"ksp_mag":KSP_Magnitude})


        self.start_time = np.array(self.orig_df["time_raw"])[0]
        self.end_time   = np.array(self.orig_df["time_raw"])[-1]
        
        #print(np.mean(Target_Flux))
        #plt.plot(BJD,Target_Flux/np.mean(self.orig_df[pd.notnull(self.orig_df["signal_raw"])]["signal_raw"]))
        #plt.show()        

    def remove_momentum_dump(self):
    
            
        if self.sector == 1:
            momentum_dump = [121,241,361,481,601,754,874,994,1114,1234,
                             88,224,340,464,546,583,731,820,854,1196,
                             749,584,225]
            self.raw_df = self.orig_df.drop(momentum_dump)
            #momentum_dump = []
            
        elif self.sector == 2:
            momentum_dump = [121,241,361,481,601,746,866,986,1106,1226,
                             338,458,573,1105,1225,1227,
                             626,642,658,674,690,706,722,738,754,770,
                             602,603,1219,1220,1221,1222,1223,1224,1225,]
            self.raw_df = self.orig_df.drop(momentum_dump)
            
        elif self.sector > 2:
            momentum_dump = np.array(self.orig_df["Flag"])
            self.raw_df = self.orig_df[self.orig_df["Flag"] != 1.]
            
        self.signal_orig = self.orig_df["signal_raw"]
        
        
        self.raw_df = self.raw_df[pd.notnull(self.raw_df["signal_raw"]) ]
        # will have to do for now.
        self.Magnitude = np.median(self.raw_df["magnitude"])
        



        
        """
        plt.plot(self.orig_df["time_raw"],self.orig_df["magnitude"])
        plt.plot(self.raw_df["time_raw"],self.raw_df["magnitude"])
        plt.show()
        
        plt.plot(self.orig_df["time_raw"],self.orig_df["signal_raw"])
        plt.plot(self.raw_df["time_raw"],self.raw_df["signal_raw"])
        plt.show()
        """
        
    def calibrate_lightcurve(self):
        
        self.time_raw = np.array(self.raw_df["time_raw"])
        self.signal_raw = np.array(self.raw_df["signal_raw"])
        self.magnitudes = np.array(self.raw_df["magnitude"])
        self.ksp_mag = np.array(self.raw_df["ksp_mag"])
        self.mag_errors = np.array(self.raw_df["mag_error"])
        self.centdx = np.array(self.raw_df["centdx"])
        self.centdy = np.array(self.raw_df["centdy"])
        
        self.time_raw_uncut = np.array(self.orig_df["time_raw"])
        self.signal_raw_uncut = np.array(self.orig_df["signal_raw"])
        self.magnitude_uncut  = np.array(self.orig_df["magnitude"])

        self.time_calibrated = self.time_raw - self.start_time
        self.time_calibrated_uncut = self.time_raw_uncut - self.start_time

        lower_threshold = np.percentile(self.signal_raw,5)
        upper_threshold = np.percentile(self.signal_raw,95)
        
        truncated_signal = self.signal_raw.copy()
        truncated_signal = truncated_signal[(self.signal_raw > lower_threshold)&(self.signal_raw < upper_threshold)]

        self.signal_calibrated = self.signal_raw/np.median(truncated_signal) + self.Normalization -1
        #self.signal_calibrated_median = self.signal_raw/np.median(truncated_signal) + self.Normalization -1
     
        self.signal_calibrated_uncut = self.signal_raw_uncut/np.median(truncated_signal) + self.Normalization -1

        # can't use this because no error found
        #self.signal_error_calibrated = self.signal_error_raw/np.median(truncated_signal)            

    def detrend_lightcurve(self,value=3):
        """
        detrending now happen on the raw data.
        Should have the ability to insert manual points for detrending
        """
        
        self.signal_detrended = []
        self.signal_cleaned   = []
        
        
        deltaT = self.time_calibrated[1:]-self.time_calibrated[:-1]        
        seperator = []
        for i in deltaT:
            if i > 0.5:  # if the gap is larger than half a day
                seperator.append(np.where(deltaT == i)[0][0]+1)
        prev = 0
        seperator.append(-1)
        for sep in seperator:
            
            if sep == -1:
                time = self.time_calibrated[prev:]
                signal = self.signal_calibrated[prev:]
            else:
                time = self.time_calibrated[prev:sep]
                signal = self.signal_calibrated[prev:sep]
                
            if len(time) == 1:
                
                if self.signal_detrended != []:
                    self.signal_detrended = np.concatenate([self.signal_detrended,[0]])
                    self.signal_cleaned     = np.concatenate([self.signal_cleaned,[0]])
                else:
                    self.signal_detrended = np.array([0])
                    self.signal_cleaned = np.array([0])
                
                
                continue
                
            
            lower_threshold = np.percentile(signal,5)
            upper_threshold = np.percentile(signal,95)
            truncated = signal.copy()
            truncated[(signal > upper_threshold)] = upper_threshold
            truncated[(signal < lower_threshold)] = lower_threshold 
            
            chunk = int((time[-2]-time[1])/value)
            
           
            if chunk == 0:
                if sep == -1:
                    tarray = [time[int((len(time)-prev)/2)]]
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
            signal_cleaned = signal_detrended.copy()
            signal_cleaned[signal_cleaned > self.std*rms] = 0
            
            if self.signal_detrended != []:
                self.signal_detrended = np.concatenate([self.signal_detrended,signal_detrended])
                self.signal_cleaned     = np.concatenate([self.signal_cleaned,signal_cleaned])
            else:
                self.signal_detrended = np.array(signal_detrended)
                self.signal_cleaned = np.array(signal_cleaned)
            
            
            """
            plt.plot(self.time_calibrated[prev:sep],signal_detrended)
            plt.plot(self.time_calibrated[prev:sep],signal_clean)
            
            # plotting the detrending
            plt.plot(self.time_calibrated[prev:sep],self.signal_calibrated[prev:sep],".")
            plt.plot(self.time_calibrated[prev:sep],self.signal_calibrated[prev:sep]-lc_filter1,"-")
            plt.plot(self.time_calibrated[prev:sep],lc_filter1,"-")
            plt.show()
            """
            if sep != -1:
                prev = sep
            else:
                prev = len(self.time_calibrated)
        
        """
        plt.plot(self.time_calibrated,self.signal_calibrated,".")
        plt.plot(self.time_calibrated,self.signal_detrended,".")
        plt.plot(self.time_calibrated,self.signal_cleaned,".")
        
        
        plt.show()
        """
           
    def bin_lightcurve(self):

        self.data_pts = int((self.end_time-self.start_time)*self.day2bin)  
        if self.data_pts%2 == 1:  # make even arrays
            self.data_pts+=1
        self.time_bin = np.arange(self.data_pts)
         
        time_floor = np.array(np.floor(self.time_calibrated*self.day2bin),dtype=int)
        self.signal_bin           = np.zeros(self.data_pts)
        self.signal_bin_detrended = np.zeros(self.data_pts)
        self.signal_bin_cleaned   = np.zeros(self.data_pts)
        self.centdx_bin           = np.zeros(self.data_pts)
        self.centdy_bin           = np.zeros(self.data_pts)
        
        self.signal_bin[time_floor] = self.signal_calibrated
        self.signal_bin_detrended[time_floor] = self.signal_detrended
        self.signal_bin_cleaned[time_floor] = self.signal_cleaned
        self.centdx_bin[time_floor] = self.centdx
        self.centdy_bin[time_floor] = self.centdy
        
        self.total_sector_span = int((self.end_time-self.start_time)/27)
        
        """
        if self.total_sector_span != 1:
            self.data_pts   = self.data_pts_per_sector
            self.time_bin   = self.time_bin[-self.data_pts:]
            self.signal_bin = self.signal_bin[-self.data_pts:]
            self.signal_bin_detrended = self.signal_bin_detrended[-self.data_pts:]
            self.signal_bin_cleaned = self.signal_bin_cleaned[-self.data_pts:]
            
            self.centdx_bin = self.centdx_bin[-self.data_pts:]
            self.centdy_bin = self.centdy_bin[-self.data_pts:]
        """

    def crop_bin_lightcurve(self,default="all"):

        if self.sector == 0:
            print("No Sector info inputed")
            sys.exit()        
        elif self.sector == 1:
            self.sector_seperator = [630,690]
            self.bad_data = [1050,1175]
            crop_location = [self.sector_seperator,self.bad_data]
        elif self.sector == 2:
            self.sector_seperator = [630,690]
            crop_location = [self.sector_seperator]
        elif self.sector == 3:
            self.sector_seperator = [600,740]
            crop_location = [[0,100],self.sector_seperator,[1150,-1]]
            
        else:
            self.sector_seperator = [650,660]
            crop_location = [self.sector_seperator]
            
        for a,b in crop_location:
            if b == -1:
                end = len(self.signal_bin)
                self.signal_bin[a:] = np.ones(end-a)*self.Normalization
                if default == "all":
                    self.signal_bin_detrended[a:] = np.ones(end-a)*self.Normalization
                    self.signal_bin_cleaned[a:] = np.ones(end-a)*self.Normalization
            elif a == 0:
                self.signal_bin[:b] = np.ones(b)*self.Normalization
                if default == "all":
                    self.signal_bin_detrended[:b] = np.ones(b)*self.Normalization
                    self.signal_bin_cleaned[:b] = np.ones(b)*self.Normalization
            else:
            
                self.signal_bin[a:b] = np.ones(b-a)*self.Normalization
                if default == "all":
                    self.signal_bin_detrended[a:b] = np.ones(b-a)*self.Normalization
                    self.signal_bin_cleaned[a:b] = np.ones(b-a)*self.Normalization

    def remove_flare(self,threshold=99):
        """
        Flares are a bit different in QLP since 30 minute is very long for many small flares
        Flares are difficult to distinguish from "jumps" in the data.
        not so much identify here anymore
        """
    
        
        upper_threshold = np.percentile(self.signal_bin,threshold)
        
        self.signal_calibrated_no_flare = self.signal_calibrated.copy()
        self.signal_calibrated_no_flare[self.signal_calibrated_no_flare > upper_threshold] = upper_threshold

        self.signal_cleaned = self.signal_bin.copy()
        self.signal_cleaned[self.signal_cleaned > upper_threshold] = upper_threshold
        


        # this does not perform well in sector 1
        # neet to manually set momentum dump location
        #filtered_df = df[df["Flag"] == b'G']

    def detrend_bin_lightcurve(self,value=3*48+1):
        
        #front,back = self.sector_seperator
        
        front = int((self.time_bin[-1]-self.time_bin[0])/2)
        back = front
        
        print(front)
        
        # detrend first orbit
        orbit1_time_bin   = np.arange(front+2*value)-value
        orbit1_signal_bin = self.signal_bin[:front]
        signal_bin = np.concatenate([np.ones(value)*orbit1_signal_bin[0],
                                     orbit1_signal_bin,
                                     np.ones(value)*orbit1_signal_bin[-1]])
        tarray = (np.arange(int(len(orbit1_time_bin)/value))+1)*value+orbit1_time_bin[0]
        tarray = tarray[:-1] # the last knot is usually too small and freaks out so remove
        lower_threshold = np.percentile(signal_bin,5)
        upper_threshold = np.percentile(signal_bin,95)
        truncated = signal_bin.copy()
        truncated[(signal_bin > upper_threshold)] = upper_threshold
        truncated[(signal_bin < lower_threshold)] = lower_threshold            
        
        sp = LSQUnivariateSpline(orbit1_time_bin, truncated,t=tarray)
        lc_filter1 = sp(orbit1_time_bin)[value:-value]
        
        # detrend second orbit
        orbit2_time_bin   = np.arange((self.data_pts-back)+2*value)-value
        orbit2_signal_bin = self.signal_bin[back:]
        signal_bin = np.concatenate([np.ones(value)*orbit2_signal_bin[0],
                                     orbit2_signal_bin,
                                     np.ones(value)*orbit2_signal_bin[-1]])
        tarray = (np.arange(int(len(orbit2_time_bin)/value))+1)*value+orbit2_time_bin[0]
        tarray = tarray[:-1] # the last knot is usually too small and freaks out so remove
        lower_threshold = np.percentile(signal_bin,5)
        upper_threshold = np.percentile(signal_bin,95)
        truncated = signal_bin.copy()
        truncated[(signal_bin > upper_threshold)] = upper_threshold
        truncated[(signal_bin < lower_threshold)] = lower_threshold            
        
        sp = LSQUnivariateSpline(orbit2_time_bin, truncated,t=tarray)
        lc_filter2 = sp(orbit2_time_bin)[value:-value]
        
        self.lc_filter = np.concatenate([lc_filter1,np.zeros(back-front),lc_filter2])
        self.signal_bin_detrended = self.signal_bin-self.lc_filter+self.Normalization
        self.signal_bin_cleaned = self.signal_cleaned-self.lc_filter+self.Normalization
        
    def remove_momentum_dump2(self,threshold=1):
        """
        Flares are a bit different in QLP since 30 minute is very long for many small flares
        Flares are difficult to distinguish from "jumps" in the data.
        not so much identify here anymore
        """
    
        lower_threshold = np.percentile(self.signal_bin,threshold)
        
        self.signal_calibrated_no_flare[self.signal_calibrated_no_flare < lower_threshold] = 0
        self.signal_cleaned[self.signal_cleaned < lower_threshold] = 0
        self.signal_bin_cleaned[self.signal_bin_cleaned < lower_threshold] = 0
        
        
class SPOC_TVOI():
    """
    TVOI: TESS Variable Object of Interest
    """
    def __init__(self,filename=None,savepath=None,sector=0,Norm=1,manual_cut=None):

        
        self.filename = filename
        self.savepath = savepath #Not used
        self.sector = sector
        self.Normalization = Norm
        self.manual_cut = manual_cut
        
        self.TIC = str(int(self.filename.split("/")[-1].split("-")[2]))
        
        self.cadence = 2.
        self.time_step = 1/720.
        self.day2bin = 24*30.
        self.bin2day = 1/self.day2bin

        self.TJD_Offset = 2457000.0
        
        self.y_unit = "ppm"
        self.pdc=False

    def load_object_data(self,detrending=721):
        
        self.load_object_information()
        self.load_object_raw_lightcurve()
        self.load_object_raw_centriod()
        self.remove_bad_lightcurve()
        self.calibrate_lightcurve()
        self.bin_lightcurve()
        self.crop_bin_lightcurve("1")
        self.identify_and_remove_flares()
        self.detrend_bin_lightcurve(detrending)
        self.crop_bin_lightcurve()
        #self.extend_bin_lightcurve()
            
    def load_object_information(self):
        
        
        # load data header
        hdul = fits.open(self.filename)
        self.hdul_header = hdul[0].header
        
        self.sector = hdul[0].header["SECTOR"]
        self.camera = hdul[0].header["CAMERA"]
        self.ccd    = hdul[0].header["CCD"]
        self.TIC_ID = hdul[0].header["TICID"]
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

    def load_object_raw_lightcurve(self):
        # load data
        self.df = Table.read(self.filename).to_pandas()  
        
        if self.sector == 4:    
            self.pdc = True  
        
        if self.pdc:
            self.filter_df = self.df[(self.df.QUALITY == 0) & 
                           (self.df.PDCSAP_FLUX != np.nan)]# bad flag filter 
            self.notnull = self.filter_df.PDCSAP_FLUX.notnull()
            self.signal_raw = np.array(self.filter_df.PDCSAP_FLUX[self.notnull])
            self.signal_error_raw = np.array(self.filter_df.PDCSAP_FLUX_ERR[self.notnull])
        else:
            self.filter_df = self.df[(self.df.QUALITY == 0) & 
                           (self.df.SAP_FLUX != np.nan)]# bad flag filter
            self.notnull = self.filter_df.SAP_FLUX.notnull()  # the indexing method for df doesn't completely remove np.nan. 
            self.signal_raw = np.array(self.filter_df.SAP_FLUX[self.notnull])
            self.signal_error_raw = np.array(self.filter_df.SAP_FLUX_ERR[self.notnull])       

        self.time_raw = np.array(self.filter_df.TIME[self.notnull])   
        self.start_time = self.df.TIME[0]

        # seperate sector into orbits by finding the index of where the data gap is
        # this needs to happen before removing bad lightcurve location
        

        
        
        if self.sector == 4:
            self.orbit_gap = self.time_raw[np.argsort(self.time_raw[1:]-self.time_raw[:-1])[-2]]
        else:
            self.orbit_gap = self.time_raw[np.argmax(self.time_raw[1:]-self.time_raw[:-1])]
        
    def load_object_raw_centriod(self):
           
        self.centdx     = self.filter_df.MOM_CENTR1[self.notnull]
        self.centdx_err = self.filter_df.MOM_CENTR1_ERR[self.notnull]
        self.centdy     = self.filter_df.MOM_CENTR2[self.notnull]
        self.centdy_err = self.filter_df.MOM_CENTR2_ERR[self.notnull]

    def remove_bad_lightcurve(self):
        """
        
        """
        
        if self.sector == 0:
            print("No Sector info inputed")
            sys.exit()
        elif self.sector == 1:
            crop_location = []#[[1,2500]]
        elif self.sector == 2:
            crop_location = []#[[1,2500]]
        elif self.sector == 3:
            crop_location = [[1,2700]]
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
            
    def calibrate_lightcurve(self,start_time=-1):
        """
        the magnitude of the target and crowdiness of the field should 
        play a role into estimating the variability of the target.
        """

        # data cleaning
        if start_time != -1:
            self.start_time = start_time
            
        self.time_calibrated = self.time_raw - self.start_time

        # but... why don't I just get the index straight away?
        self.orbit_gap_calibrated = self.orbit_gap - self.start_time
        self.orbit_gap_index = np.where(self.time_calibrated == self.orbit_gap_calibrated)[0][0]+1


        lower_threshold = np.percentile(self.signal_raw,5)
        upper_threshold = np.percentile(self.signal_raw,95)
        truncated_signal = self.signal_raw.copy()
        truncated_signal = truncated_signal[(self.signal_raw > lower_threshold)&(self.signal_raw < upper_threshold)]

        self.signal_calibrated = self.signal_raw/np.median(truncated_signal) + self.Normalization -1
        #self.signal_calibrated_median = self.signal_raw/np.median(truncated_signal) + self.Normalization -1
     
        self.signal_calibrated_uncut = self.signal_raw_uncut/np.median(truncated_signal) + self.Normalization -1
     
        self.signal_error_calibrated = self.signal_error_raw/np.median(truncated_signal)
        
    def bin_lightcurve(self):
        """
        There is a index problem here that may effect binning result
        
        bin before split lightcurve
        
        """

        self.data_pts = int((self.time_calibrated[-1])*self.day2bin)  
        
        self.data_pts+=1
            
        self.time_bin = np.arange(self.data_pts)
        
        """
        y_interp = interp1d(self.time_calibrated*self.day2bin, 
                            self.signal_calibrated, 
                            fill_value="extrapolate")
        
        self.signal_bin = y_interp(self.time_bin)
        """
         
        time_floor = np.array(np.floor(self.time_calibrated*self.day2bin),dtype=int)
        self.signal_bin           = np.zeros(self.data_pts)
        self.signal_bin[time_floor] = self.signal_calibrated
         
 
 
 
 
        
        centdx_interp = interp1d(self.time_calibrated*self.day2bin, 
                            self.centdx, 
                            fill_value="extrapolate")
        
        self.centdx_bin = centdx_interp(self.time_bin)
        
        centdy_interp = interp1d(self.time_calibrated*self.day2bin, 
                            self.centdy, 
                            fill_value="extrapolate")
        
        self.centdy_bin = centdy_interp(self.time_bin)        
        
        
        error_interp = interp1d(self.time_calibrated*self.day2bin,
                                self.signal_error_calibrated,
                                fill_value="extrapolate")
        
        self.signal_error_bin = error_interp(self.time_bin)        


        
        """
        
        #self.gap_time_bin = self.time_bin[orbit1_bin_end:orbit2_bin_start]
        #self.gap_signal_bin = np.zeros(orbit2_bin_start-orbit1_bin_end)
        orbit1_start = 0
        orbit1_end   = self.time_calibrated[self.orbit_gap_index-1]
        orbit2_start = self.time_calibrated[self.orbit_gap_index]
        orbit2_end   = self.time_raw[-1]-self.start_time
        
        self.orbit1_time_bin = np.arange(int((orbit1_end-orbit1_start)*self.day2bin))
        self.gap_time_bin    = np.arange(int((orbit2_start-orbit1_end)*self.day2bin))+self.orbit1_time_bin[-1]
        self.orbit2_time_bin = np.arange(int((orbit2_end-orbit2_start)*self.day2bin))+self.gap_time_bin[-1]

        print(self.start_time,self.time_raw[-1]-self.start_time)
        print(len(self.time_bin))
        print(len(self.orbit1_time_bin),len(self.orbit2_time_bin),len(self.gap_time_bin))


        y_interp = interp1d(self.time_calibrated[:self.orbit_gap_index]*self.day2bin, 
                            self.signal_calibrated[:self.orbit_gap_index], 
                            fill_value="extrapolate")
        
        self.orbit1_signal_bin = y_interp(self.orbit1_time_bin)

        y_interp = interp1d(self.time_calibrated[self.orbit_gap_index-1:self.orbit_gap_index+1]*self.day2bin, 
                            self.signal_calibrated[self.orbit_gap_index-1:self.orbit_gap_index+1], 
                            fill_value="extrapolate")
        
        self.gap_signal_bin = y_interp(self.gap_time_bin)


        y_interp = interp1d(self.time_calibrated[self.orbit_gap_index:]*self.day2bin, 
                            self.signal_calibrated[self.orbit_gap_index:], 
                            fill_value="extrapolate")
        
        self.orbit2_signal_bin = y_interp(self.orbit2_time_bin)

        print(len(self.orbit1_signal_bin)+len(self.orbit2_signal_bin)+len(self.gap_time_bin))
        
        plt.plot(self.orbit1_time_bin,self.orbit1_time_bin,".")
        plt.plot(self.gap_time_bin[1:-1],self.gap_time_bin[1:-1],".")
        plt.plot(self.orbit2_time_bin,self.orbit2_time_bin,".")
        plt.show()


        """

    def identify_and_remove_flares(self):
        """
        remove flare before or after detrending?
        
        This flare identifier is not good, replace with a box car instead
        does not perform well for some very varying stars
        """
        TESSMAG = self.object["TESSMAG"]
        self.flare_location,self.peak_threshold,self.wing_threshold = LCA.find_flares(self.signal_bin,TESSMAG)
        self.num_flares = len(self.flare_location)
        
        
        if len(self.flare_location) == 0:
            self.signal_cleaned = self.signal_bin
            self.signal_calibrated_no_flare = self.signal_calibrated
        else:
            # percentile of the top most 100 data point? 
            # I don't understand why I'm doing this... need to change
            # kinda work but makes no sense
            # Also why flare removal need to happen before binning?? 
            # changed flare removal to happen after binning
            # also flare location would be good indicator for where to remove flares.... 
            # flare energy study needed here too....
            # update flare with kepler paper in the future
            
            # i shouldn't be randomly cutting data if there are not points above
            flare_pts = len(self.flare_location)*30
            a = (len(self.signal_bin)-flare_pts)/len(self.signal_bin)*100
            percent = a if a > 99 else 99
            
            upper_threshold = np.percentile(self.signal_bin,percent)
            
            
            
            self.signal_calibrated_no_flare = self.signal_calibrated.copy()
            self.signal_calibrated_no_flare[self.signal_calibrated_no_flare > upper_threshold] = upper_threshold
            
            self.signal_cleaned = self.signal_bin.copy()
            self.signal_cleaned[self.signal_cleaned > upper_threshold] = upper_threshold

    def detrend_bin_lightcurve(self,value=3*720+1):
        """
        pad the array by 720 bins in each direction 
        I might need to have multiple layer of detrending
        """
        
        time_bin   = np.arange(self.data_pts+2*value)-value
        signal_bin = np.concatenate([np.ones(value)*self.signal_bin[0],
                                     self.signal_bin,
                                     np.ones(value)*self.signal_bin[-1]])
        
        tarray = (np.arange(int(len(time_bin)/value))+1)*value+time_bin[0]

        tarray = tarray[:-1] # the last know is usually too small and freaks out
        
        exclude = 90
        lower_threshold = np.percentile(signal_bin,50-exclude/2.)
        upper_threshold = np.percentile(signal_bin,50+exclude/2.)
        
        truncated = signal_bin.copy()
        truncated[(signal_bin > upper_threshold)] = upper_threshold
        truncated[(signal_bin < lower_threshold)] = lower_threshold            
        
        sp = LSQUnivariateSpline(time_bin, truncated,t=tarray)
        #sp.set_smoothing_factor(10)

        self.lc_filter = sp(time_bin)[value:-value]
        
        self.signal_bin_detrended = self.signal_bin-self.lc_filter+self.Normalization
        self.signal_bin_cleaned = self.signal_cleaned-self.lc_filter+self.Normalization

    def crop_bin_lightcurve(self,default="all"):

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
        
        if self.manual_cut == []:
            pass
        else:
            crop_location = self.manual_cut        
        
        for a,b in crop_location:
            if b == -1:
                end = len(self.signal_bin)
                self.signal_bin[a:] = np.ones(end-a)*self.Normalization
                if default == "all":
                    self.signal_bin_detrended[a:] = np.ones(end-a)*self.Normalization
                    self.signal_bin_cleaned[a:] = np.ones(end-a)*self.Normalization
            elif a == 0:
                self.signal_bin[:b] = np.ones(b)*self.Normalization
                if default == "all":
                    self.signal_bin_detrended[:b] = np.ones(b)*self.Normalization
                    self.signal_bin_cleaned[:b] = np.ones(b)*self.Normalization
            else:
            
                self.signal_bin[a:b] = np.ones(b-a)*self.Normalization
                if default == "all":
                    self.signal_bin_detrended[a:b] = np.ones(b-a)*self.Normalization
                    self.signal_bin_cleaned[a:b] = np.ones(b-a)*self.Normalization
                    
    def load_lightcurve(self,datapath):
        """
        load preprocessed lightcurve
        
        The loaded lightcurve will not be extended to save space
        The time bin can also be recalculated from load. 
        
        """
        
        self.signal_bin_filtered = np.load(datapath)
        self.time_bin_extended = np.arange(self.num_bin)
    
    def save_lightcurve(self,outputpath):
        """
        only saving the signal_bin_filtered because it can be easily extended
        time array don't need to be pre-calculated for ever object. It's also same for every object
        """
        
        np.save(outputpath,self.signal_bin_filtered)
        
    def inject_fake_signal(self,replace=True):
        
        
        # inject a fake signal at 200 cycle/day
        # I assume the amplitude injected here should scale with the noise 
        # But I won't know the noise until i do fft analysis.
        # is it possible to know this from the raw data?
        # perhaps I can calculate a rms for the raw data?
        # 1% the rms of raw data? or ... 1% of max-min? what should the 1%median        
        if replace:
            self.signal_bin_filtered = 100/1e6*np.sin(200./360*np.pi*self.time_bin)
        else:
            self.signal_bin_filtered_fake = 100/1e6*np.sin(200./360*np.pi*self.time_bin)
                
        



if __name__ == "__main__":
    
    filename = "test_data/206544316.h5"
    TVOI = QLP_TVOI(filename,sector=1)
    TVOI.load_object_data()




