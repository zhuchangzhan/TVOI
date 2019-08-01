"""

I've gotten to a point where the Lightcurve_io need a complete rewrite


What really defines a "lightcurve" object? The current class is too ambiguous about what it does

I think what I really should do is to have a TOI object.
Under each TOI object is data from each sector/orbit if available.
both sector and orbit data are technically lightcurves but they themself contain more info than the lightcurve.

lightcurve should really just be the x and y data.
In the future, The processor will be TOI based and each TOI will try to collect data from as many sector as possible
For now constrained to each sector, but break into multiple orbits. 


detrending should happen on the orbit level

ultimately this will require a database containing basic information about each TOI object so 
I don't have to dig it up everytime. But... for now this will do.





Tasks:
    1. store each orbit individually
    2. move detrending to LC_Analysis
    3. need to double check if I'm making .copy or not
    4. redo inject signal


"""
import os
import sys
import numpy as np

from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel

DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(DIR, '../..'))

import src.main.LC_Analysis as LC


class TOI():
    """
    Should think about implementing inheritance in the future
    """
    
    def __init__(self):
        """
        
        """
        pass

class QLP_TOI():
    
    def __init__(self):
        pass

class SPOC_TOI():
    """
    TOI is probably an unofficial name until I know the exact terminology
    """
    def __init__(self,
                 TIC=None,
                 Target_Sector="-1",
                 filepath=None,
                 savepath=None,
                 from_file=False,
                 generate_save=True):
        """
        TOI can be called from both a TIC ID or filepath.
        Right now I'm working with a filepath but in the future this will 
        more likely to be called from a TIC ID.
        
        cadence and num_bin is not really used in version one since I'm only
        dealing with SPOC data. I should write another wrapper to deal with qlp data.
        At present stage it's not wise to create a "do all" processor.
        
        TOI: TIC ID, usually a int or string, both is fine
        filepath: Path to data
        savepath: Location to store output data.
        
        """
        self.TIC = TIC
        self.Target_Sector = Target_Sector
        self.filepath = filepath
        self.savepath = savepath
        self.from_file = from_file
        
        self.cadence = 2.
        self.time_step = 1/720.
        self.day2bin = 24*30.
        self.bin2day = 1/self.day2bin
        
        if savepath!= None and generate_save and not os.path.isdir(savepath):
            print("Creating Path: %s"%savepath)
            os.makedirs(savepath)             
        
        self.TJD_Offset = 2457000.0
        self.available_sector = ["1","2","3","4"]
        self.sectors = {}
        for sec in self.available_sector:
            self.sectors[sec] = []
        
        if self.Target_Sector == "-1":
            print("Please specify which sector to load")
        elif self.Target_Sector == "0":
            print("Load all sector data, not implemented")
        elif self.Target_Sector not in self.available_sector:
            print("Sector not available.")
        elif type(self.Target_Sector) == list:
            print("Load multi_sector, not implemented")
        else:
            self.sectors[self.Target_Sector] = Sector(TOI,Target_Sector,filepath,savepath,from_file) 

    def load_lightcurve_data(self,y_unit="percent",filter_method="spline",filter_value=240):

        # for future multi sector processing
        for sec in self.available_sector:
            pass
        
        cur_Sec = self.sectors[self.Target_Sector]
        cur_Sec.load_sector_data(y_unit=y_unit) 
        
        self.object = cur_Sec.object
        
        orbit1 = cur_Sec.Orbit1
        orbit2 = cur_Sec.Orbit2
        
        orbit1.signal_bin_filter = LC.detrend_lightcurve(orbit1.time_bin,
                                                         orbit1.signal_bin,
                                                         method=filter_method,
                                                         value=filter_value)
        
        orbit2.signal_bin_filter = LC.detrend_lightcurve(orbit2.time_bin,
                                                         orbit2.signal_bin,
                                                         method=filter_method,
                                                         value=filter_value)
        
        orbit1.signal_bin_filtered = orbit1.signal_bin - orbit1.signal_bin_filter
        orbit2.signal_bin_filtered = orbit2.signal_bin - orbit2.signal_bin_filter
        
        orbit1.signal_bin_filtered_truncated = orbit1.signal_bin_flare_truncated - orbit1.signal_bin_filter
        orbit2.signal_bin_filtered_truncated = orbit2.signal_bin_flare_truncated - orbit2.signal_bin_filter
        
        
        
        cur_Sec.signal_flare_truncated    = np.concatenate([orbit1.signal_flare_truncated,
                                                      orbit2.signal_flare_truncated])      

        cur_Sec.signal_calibrated   = np.concatenate([orbit1.signal_calibrated,
                                                      orbit2.signal_calibrated])  

        cur_Sec.signal_bin          = np.concatenate([orbit1.signal_bin,
                                                      cur_Sec.gap_signal_bin,
                                                      orbit2.signal_bin])  
              
        cur_Sec.signal_bin_filter   = np.concatenate([orbit1.signal_bin_filter,
                                                      cur_Sec.gap_signal_bin,
                                                      orbit2.signal_bin_filter])   
        
        cur_Sec.signal_bin_filtered = np.concatenate([orbit1.signal_bin_filtered,
                                                      cur_Sec.gap_signal_bin,
                                                      orbit2.signal_bin_filtered])
        
        cur_Sec.signal_bin_filtered_truncated = np.concatenate([orbit1.signal_bin_filtered_truncated,
                                                      cur_Sec.gap_signal_bin,
                                                      orbit2.signal_bin_filtered_truncated])
        
        
        
        cur_Sec.time_flare_truncated = np.concatenate([orbit1.time_flare_truncated,
                                                       orbit2.time_flare_truncated])  

        
        
        # also need to concatenate the sector gaps in the future for multi-sector processing
        self.start_time = cur_Sec.start_time
        self.time_bin = cur_Sec.time_bin
        self.time_calibrated = cur_Sec.time_calibrated
        self.time_raw = cur_Sec.time_raw
        self.signal_raw = cur_Sec.signal_raw
        self.signal_bin = cur_Sec.signal_bin
        self.signal_calibrated = cur_Sec.signal_calibrated
        self.signal_bin_filter = cur_Sec.signal_bin_filter
        self.signal_bin_filtered = cur_Sec.signal_bin_filtered
        self.signal_flare_truncated = cur_Sec.signal_flare_truncated
        self.time_flare_truncated = cur_Sec.time_flare_truncated
        self.signal_bin_filtered_truncated = cur_Sec.signal_bin_filtered_truncated

        self.orbit1 = orbit1
        self.orbit2 = orbit2

class Sector():
    
    def __init__(self,TIC=None,Target_Sector="-1",filepath=None,savepath=None,from_file=False):
        
        self.TIC = TIC
        self.Target_Sector = Target_Sector
        self.filepath = filepath
        self.savepath = savepath
        self.from_file = from_file
        
    def load_sector_data(self, y_unit="percent",pdc=False):
        
        self.y_unit = y_unit

        if self.filepath != None:
            if self.from_file: # assuming the output file is a numpy file. 
                self.time_calibrated, self.signal_calibrated = np.load(self.filepath)
                return
            #self.load_data()
        elif self.TIC != None:
            print("Not Implemented. In the future will search for existing data. Need database")
        else:
            print("Not valid input")
            return

        if type(self.filepath) == list:
            print("Not implemented, uses first item in filepath. In the future this will process multisector data")
            self.filepath = self.filepath[0]
        
        # load data header
        hdul = fits.open(self.filepath)
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

        # load data
        df = Table.read(self.filepath).to_pandas()        
        

        
        if pdc:
            filter_df = df[(df.QUALITY == 0) & 
                           (df.PDCSAP_FLUX != np.nan)]# bad flag filter 
            notnull = filter_df.PDCSAP_FLUX.notnull()
            self.signal_raw = np.array(filter_df.PDCSAP_FLUX[notnull])
            self.signal_error_raw = np.array(filter_df.PDCSAP_FLUX_ERR[notnull])
        else:
            filter_df = df[(df.QUALITY == 0) & 
                           (df.SAP_FLUX != np.nan)]# bad flag filter
            notnull = filter_df.SAP_FLUX.notnull()  # the indexing method for df doesn't completely remove np.nan. 
            self.signal_raw = np.array(filter_df.SAP_FLUX[notnull])
            self.signal_error_raw = np.array(filter_df.SAP_FLUX_ERR[notnull])       

             
        self.time_raw = np.array(filter_df.TIME[notnull])            
        self.centdx     = filter_df.MOM_CENTR1[notnull]
        self.centdx_err = filter_df.MOM_CENTR1_ERR[notnull]
        self.centdy     = filter_df.MOM_CENTR2[notnull]
        self.centdy_err = filter_df.MOM_CENTR2_ERR[notnull]     



        
        # data cleaning
        self.start_time = self.time_raw[0] # convert this to utc?
        self.time_calibrated = self.time_raw-self.time_raw[0]

        # calculate time bin. This should move to SPOC level once we start to play with multisector data
        self.day2bin = 24*30.
        self.time_bin = np.arange(int((self.time_calibrated[-1]-self.time_calibrated[0])*self.day2bin))      
        
        # seperate sector into orbits
        time_diff = self.time_calibrated[1:]-self.time_calibrated[:-1]
        self.orbit_gap = np.argmax(time_diff)+1
        
        # bin the data. Data binning needs to happen in the SPOC level to align all data to the same time bin
        orbit1_bin_start = 0
        orbit1_bin_end = int(self.time_calibrated[self.orbit_gap-1]*self.day2bin)
        orbit2_bin_start = int(self.time_calibrated[self.orbit_gap]*self.day2bin)  
        #orbit2_bin_end = -1    # this will cause the last data point to be missing
        
        self.gap_time_bin = self.time_bin[orbit1_bin_end:orbit2_bin_start]
        self.gap_signal_bin = np.zeros(orbit2_bin_start-orbit1_bin_end)
        
        
        
        """
        print(len(self.time_bin))
        
        orbit1_time_bin = np.arange(int((self.time_calibrated[self.orbit_gap-1]-self.time_calibrated[0])*self.day2bin))
        orbit2_time_bin = np.arange(int((self.time_calibrated[-1]-self.time_calibrated[self.orbit_gap])*self.day2bin))
        gap_time_bin = np.arange(int((self.time_calibrated[self.orbit_gap]-self.time_calibrated[self.orbit_gap-1])*self.day2bin))
        
        print(len(orbit1_time_bin)+len(orbit2_time_bin)+len(gap_time_bin))
        """
        
        # load data into orbit where detrending will happen
        self.Orbit1 = Orbit(self.time_calibrated[:self.orbit_gap],
                            self.signal_raw[:self.orbit_gap],
                            self.y_unit,
                            self.day2bin,
                            self.time_bin[orbit1_bin_start:orbit1_bin_end])
        
        self.Orbit2 = Orbit(self.time_calibrated[self.orbit_gap:],
                            self.signal_raw[self.orbit_gap:],
                            self.y_unit,
                            self.day2bin,
                            self.time_bin[orbit2_bin_start:])
        
        # stitch orbit data together? This should happen on the TOI level though. 
           
        


class Orbit():
    
    def __init__(self,time,signal,y_unit,day2bin,time_bin):
        
        self.time_calibrated = time
        self.signal_raw = signal
        self.y_unit = y_unit
        self.day2bin = day2bin
        self.time_bin = time_bin
        self.dp = len(self.signal_raw)
    
    
        if self.y_unit == "flux":
            signal_calibrated = self.signal_raw
        
        elif self.y_unit == "percent":
            
            # Calculate mean of the data with 90 percentile
            lower_threshold = np.percentile(self.signal_raw,5)
            upper_threshold = np.percentile(self.signal_raw,95)
            truncated_signal = self.signal_raw.copy()
            truncated_signal = truncated_signal[(self.signal_raw > lower_threshold)&(self.signal_raw < upper_threshold)]
            mean = np.mean(truncated_signal)
            self.signal_calibrated = self.signal_raw/mean - 1
    
        else:
            print("y unit Not Implemented")
    
        self.remove_flare()
        self.bin_data()
        


    def remove_flare(self):
        
        
        a = (self.dp-100)/self.dp*100
        
        percent = a if a > 95 else 95
        
        
        #lower_threshold = np.percentile(self.signal_bin,50-exclude/2.)
        upper_threshold = np.percentile(self.signal_calibrated,percent)
        
        
        self.signal_flare_truncated = self.signal_calibrated.copy()
        self.time_flare_truncated   = self.time_calibrated.copy()
        
        self.signal_flare_truncated = self.signal_flare_truncated[self.signal_calibrated < upper_threshold]
        self.time_flare_truncated = self.time_flare_truncated[self.signal_calibrated < upper_threshold]
        
        """
        import matplotlib.pyplot as plt
        plt.plot(self.time_calibrated,self.signal_calibrated)
        plt.plot(self.time_flare_truncated,self.signal_flare_truncated)
        plt.show()
        print(len(self.time_flare_truncated),len(self.time_calibrated))
        """
        
    def bin_data(self):
        """
        There may be some problem with binning data for each orbit to be not compatible.
        
        """
    
        y_interp = interp1d(self.time_calibrated*self.day2bin, 
                            self.signal_calibrated, 
                            fill_value="extrapolate")
        
        self.signal_bin = y_interp(self.time_bin)
        
        t_interp = interp1d(self.time_flare_truncated*self.day2bin, 
                            self.signal_flare_truncated, 
                            fill_value="extrapolate")
                
        self.signal_bin_flare_truncated = t_interp(self.time_bin) 
        
        """
        import matplotlib.pyplot as plt
        plt.plot(self.time_flare_truncated*self.day2bin, 
                            self.signal_flare_truncated)
        plt.plot(self.time_bin, self.signal_bin_flare_truncated)
        plt.show()
        """
        
        
