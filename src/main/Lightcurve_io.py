# -*- encoding: utf-8 -*-
"""

need to double check if I'm making .copy or not

"""
import sys
import numpy as np

from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel


def crop_data(inputarray,crop_location,crop_value = 0):
    """
    remove unnecessary data by setting it to zero without touching the base data
    should this be a TOI method?
    """
    
    newarray = inputarray.copy()
    
    for loc in crop_location:
        a,b = loc
        if crop_value == 0:
            newarray[a:b] = np.zeros(b-a)
        elif crop_value == np.nan:
            newarray[a:b] = [np.nan]*(b-a)
        else:
            newarray[a:b] = np.ones(b-a)*crop_value
            # how to make a nan array?
            
    return newarray

class lightcurve():
    """
    Creating a light curve class to handle data io
    pertain to that of lightcurves. 
    
    Calculation and transformation will happen to the lightcurve object
    It will hold many status and stages of products
    
    """
    
    def __init__(self,
                 target=None,
                 cadence=None,
                 filepath=None,
                 savepath=None,
                 y_unit="flux",
                 inject=False,
                 num_bin=None,
                 ):
        """
        The lightcurve object is initiated with a TIC id in most scenarios.
        
        I can implement other search mechanism if needed.  
        """
        
        self.target = target
        self.cadence = cadence
        self.filepath = filepath
        self.savepath = savepath
        self.y_unit = y_unit
        self.inject = inject

        if self.cadence == None:
            print("Cadence unspecified")
            raise ValueError      

        # need a smarter way to find roundly 8x of data
        # this will have to change when we work with multiple sectors.
        # convert unit of data from days to 2-minute.     
        # 30 days * 24 hours in 1 day * 30 2-minute in 1 hour
        if num_bin == None:
            if self.cadence == 30:
                self.num_bin = 8192.
                self.day2bin = 24*2.
            elif self.cadence == 2:
                self.num_bin = 2**17
                self.day2bin = 24*30.
        else:
            self.num_bin = num_bin

        if self.cadence == 2 :
            self.time_step = 1/720.
        elif self.cadence == 30:
            self.time_step = 1/48.


    def get_lightcurve_long(self, filename, filepath):
        """
        Load in long cadence data
        Not needed for now.
        Will be implemented in future rewrites of the code
        """
        pass

    def get_lightcurve_short(self,from_file=False,pdc=False):
        """
        load in short cadence data
        
        """
        
        if from_file:
            self.time_calibrated, self.signal_calibrated = np.load(self.filepath)
            return
        
        
        hdul = fits.open(self.filepath)
        #for i in hdul[0].header:
        #    print(i, hdul[0].header[i])
        
        
        self.sector = hdul[0].header["SECTOR"]
        self.camera = hdul[0].header["CAMERA"]
        self.ccd    = hdul[0].header["CCD"]
        
        self.TIC_ID = hdul[0].header["TICID"]

        self.Tstart = hdul[0].header["TSTART"]
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





        

        TJD_Offset = "2457000"
        

        df = Table.read(self.filepath).to_pandas()
        
        #print(df.dtypes.index)
        #sys.exit()
        
        
        # The data is still problematic and some region need to be manually excluded

        if pdc:
            filter_df = df[(df.QUALITY == 0) & 
                           (df.PDCSAP_FLUX != np.nan)]# bad flag filter 
            notnull = filter_df.PDCSAP_FLUX.notnull()
        else:
            filter_df = df[(df.QUALITY == 0) & 
                           (df.SAP_FLUX != np.nan)]# bad flag filter
            notnull = filter_df.SAP_FLUX.notnull()  # the indexing method for df doesn't completely remove np.nan. 
        
        self.time_raw = np.array(filter_df.TIME[notnull])
        
        #print(filter_df.SAP_FLUX.isnull().sum())

        try:
            self.centdx     = filter_df.MOM_CENTR1[notnull]
            self.centdx_err = filter_df.MOM_CENTR1_ERR[notnull]
            self.centdy     = filter_df.MOM_CENTR2[notnull]
            self.centdy_err = filter_df.MOM_CENTR2_ERR[notnull]
        except:
            pass
        
        if self.y_unit == "flux":
            if pdc:
                self.signal_raw = np.array(filter_df.PDCSAP_FLUX[notnull])
                self.signal_error_raw = np.array(filter_df.PDCSAP_FLUX_ERR[notnull])
            else:
                self.signal_raw = np.array(filter_df.SAP_FLUX[notnull])
                self.signal_error_raw = np.array(filter_df.SAP_FLUX_ERR[notnull])
                
                
        elif self.y_unit == "ppm":
            if pdc:
                self.signal_raw = np.array(filter_df.PDCSAP_FLUX[notnull])
                self.signal_error_raw = np.array(filter_df.PDCSAP_FLUX_ERR[notnull])
            else:
                self.signal_raw = np.array(filter_df.SAP_FLUX[notnull])
                self.signal_error_raw = np.array(filter_df.SAP_FLUX_ERR[notnull])
        else:
            print("%s not implemented"%self.y_unit)
            sys.exit()
        
        
        
        
        # need a way to automatically detect large data gaps and assess right values to put in.
        # need to figure out how big are the data gaps and know when to append and when to ignore.
        # large gaps are likely due to downlink. smaller gaps could due to momentum dump
        # may require a list that contains documentation on when downlink or momentum dumps happen. 
        self.start_time = self.time_raw[0] # convert this to utc?
        self.time_calibrated = self.time_raw-self.time_raw[0]
        
        if pdc:
            self.flux_calibrated = self.signal_raw-filter_df.PDCSAP_FLUX.mean()
            self.ppm_calibrated = self.signal_raw/filter_df.PDCSAP_FLUX.mean()-1        
        else:
            self.flux_calibrated = self.signal_raw-filter_df.SAP_FLUX.mean()
            self.ppm_calibrated = self.signal_raw/filter_df.SAP_FLUX.mean()-1

        self.raw_mag = [] # not something the SPOC data provides? need to look into this if needed.
        self.calibrated_mag = [] # will implement this when needed.
        
        if self.y_unit == "flux":
            self.signal_calibrated = self.flux_calibrated
        elif self.y_unit == "mag":
            self.signal_calibrated = self.mag_calibrated
        elif self.y_unit == "ppm":
            self.signal_calibrated = self.ppm_calibrated


        self.signal_error_calibrated = self.signal_error_raw/filter_df.SAP_FLUX.mean()





    def bin_lightcurve(self):
        """
        interpolate the data into equal length bins 
    
        Each bin is 2 minute (SPOC) or 30 minute (FFI) or arb. minute for future data

        signal will be automatically extended to roughly 8 times when doing the fft, 
        there is no need for me to do this at the binning stage.    
        """
        # something needs to be done here to make the bins better
        self.data_pts = int((self.time_calibrated[-1]-self.time_calibrated[0])*self.day2bin)
        self.time_bin = np.arange(self.data_pts)
        
        y_interp = interp1d(self.time_calibrated*self.day2bin, 
                            self.signal_calibrated, 
                            fill_value="extrapolate")
        
        self.signal_bin = y_interp(self.time_bin)
        
        
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
        
        
        
    def filter_lightcurve(self, method="boxcar",value=721):
        """
        filter the lightcurve data
        
        need to work on a spline filter
        
        The order I think should be: bin, filter, inject, extend. Crop will be a function independent of order 
        """
        
        self.filter_method = method

        if self.filter_method == "boxcar":
            self.lc_filter = convolve(self.signal_bin, Box1DKernel(value)) # the "thing" to subtract
        elif self.filter_method == "spline":
            from scipy.interpolate import LSQUnivariateSpline 
            
            # how do we handle the last knot point? it's usually shorter than the rest
            self.tarray = (np.arange(int(self.data_pts/value))+1)*value
            
            
            exclude = 90
            lower_threshold = np.percentile(self.signal_bin,50-exclude/2.)
            upper_threshold = np.percentile(self.signal_bin,50+exclude/2.)
            
            truncated = self.signal_bin.copy()
            truncated[(self.signal_bin > upper_threshold)] = upper_threshold
            truncated[(self.signal_bin < lower_threshold)] = lower_threshold            
            
            
            
            
            
            
            sp = LSQUnivariateSpline(self.time_bin, truncated,t=self.tarray)
            sp.set_smoothing_factor(10)
            self.lc_filter = sp(self.time_bin)
            
        else:
            print("Not implemented")
            sys.exit()
            
        self.signal_bin_filtered = self.signal_bin - self.lc_filter
    
    def inject_fake_signal(self,replace=True):
        
        
        # inject a fake signal at 200 cycle/day
        # I assume the amplitude injected here should scale with the noise 
        # But I won't know the noise until i do fft analysis.
        # is it possible to know this from the raw data?
        # perhaps I can calculate a rms for the raw data?
        # 1% the rms of raw data? or ... 1% of max-min? what should the 1%mean        
        if replace:
            self.signal_bin_filtered = 100/1e6*np.sin(200./360*np.pi*self.time_bin)
        else:
            self.signal_bin_filtered_fake = 100/1e6*np.sin(200./360*np.pi*self.time_bin)
        
    def extend_lightcurve(self):

        # Each bin is 2 minute.  a weird unit to be for sure.
        # need to work this into Hz, cycle/day, frequency. puzzling
        self.time_bin_extended = np.arange(self.num_bin) # unit: 2-minute
        
        self.signal_bin_filtered_extended = np.concatenate([self.signal_bin_filtered,
                                                            np.zeros(self.num_bin-self.data_pts)])
        
        # only use for folding data?
        self.time_bin_day = self.time_bin/self.day2bin
        self.time_bin_extended_day = self.time_bin_extended/self.day2bin 
    
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
        
        
        
        
        
