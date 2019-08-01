"""

simple tools to get data for the object of interest

April 2019:
Updated to look for QLP LC

"""
import h5py
import os,sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.table import Table


DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(DIR, '../..'))

from src.util.common_filepaths import return_single_target_datapath

import matplotlib.pyplot as plt

def get_flux_QLP(sector,TIC,filepath=None,pdc=False):
    
    if filepath == None:
        filepath,cam,ccd = return_single_target_datapath(sector,TIC,"QLP")    
        
    if filepath == "Not found":
        return [],[],0,0,filepath 
    print(filepath)
    file = h5py.File(filepath, 'r')
    
    mag2flux_0 = 1.48*10e7
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

    # also how do you get error on the flux? 
    Target_Magnitude = np.array(AperturePhotometry["Aperture_002"]["RawMagnitude"])

    # Convert magnitude to flux
    Target_Flux = 10**(-Target_Magnitude/2.5)*mag2flux_0
    
    if sector > 1:
        QualityFlag = Lightcurve["QFLAG"]
    else:
        QualityFlag = AperturePhotometry["Aperture_002"]["QualityFlag"]

    orig_df = pd.DataFrame({"Flag":QualityFlag,"time_raw":BJD,"signal_raw":Target_Flux,
                           "centdx":Centroid_X,"centdy":Centroid_Y,"magnitude":Target_Magnitude})

    if sector == 1:
        momentum_dump = [121,241,361,481,601,754,874,994,1114,1234,
                         88,224,340,464,546,583,731,820,854,1196,
                         749,584,225]
        raw_df = orig_df.drop(momentum_dump)
    else:
        momentum_dump = np.array(orig_df["Flag"])
        raw_df = orig_df[orig_df["Flag"] != 1.]
    
    raw_df = raw_df[pd.notnull(raw_df["signal_raw"])]

    time_raw = raw_df["signal_raw"]
    signal_raw = raw_df["time_raw"]

    return time_raw,signal_raw,cam,ccd,filepath

def get_flux_SPOC(sector,TIC,filepath=None,pdc=False):


    if filepath == None:
        filepath = return_single_target_datapath(sector,TIC)
        
    if filepath == "Not found":
        return [],[],filepath
    
    #print(filepath)
    try:
        hdul = fits.open(filepath)
    except:
        return [],[],"Not found"
    
    df = Table.read(filepath).to_pandas()
    if pdc:
        filter_df = df[(df.QUALITY == 0) & 
                       (df.PDCSAP_FLUX != np.nan)]# bad flag filter 
        notnull = filter_df.PDCSAP_FLUX.notnull()
        signal_raw = np.array(filter_df.PDCSAP_FLUX[notnull])
    else:
        filter_df = df[(df.QUALITY == 0) & 
                       (df.SAP_FLUX != np.nan)]# bad flag filter
        notnull = filter_df.SAP_FLUX.notnull()  # the indexing method for df doesn't completely remove np.nan. 
        signal_raw = np.array(filter_df.SAP_FLUX[notnull])   

    time_raw = np.array(filter_df.TIME[notnull])  
    
    """
    day2bin = 1./720
    time_calibrated = time_raw-time_raw[0]
    signal_calibrated = signal_raw/np.mean(signal_raw) -1
    
    data_pts = int((time_raw[-1]-time_raw[0])*day2bin)  
    time_bin = np.arange(data_pts)
    y_interp = interp1d(time_calibrated*day2bin, 
                        signal_calibrated, 
                        fill_value="extrapolate")
    signal_bin = y_interp(time_bin)
    """
    return time_raw,signal_raw,filepath#,signal_calibrated,time_bin,signal_bin,filepath

def convert_fits_to_txt(sector,TIC,filepath=None,savepath=None,pdc=False,verbose=False):
    
    data = get_flux_SPOC(sector,TIC,filepath,pdc)
    
    time_raw,signal_raw,filepath = data
    
    if filepath == "Not found":
        if verbose:
            print("%s not found in SPOC sector %s"%(TIC,sector))
        return
    else:
        print("SPOC",filepath)
    
    if savepath == None:
        savepath = "output"
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 

    with open("%s/SPOC_%s_%s_raw.txt"%(savepath,sector,TIC),"w") as f:
        for i,j in zip(time_raw,signal_raw):
            f.write(",".join([str(i),str(j),"\n"]))
    """
    with open("%s/%s_bin.txt"%(savepath,TIC),"w") as f:
        for i,j in zip(time_bin,signal_bin):
            f.write(" ".join([str(i),str(j),"\n"]))    
    """
    return

def convert_h5_to_txt(sector,TIC,filepath=None,savepath=None,verbose=False):

    data = get_flux_QLP(sector,TIC,filepath)
    time_raw,signal_raw,cam,ccd,filepath = data
    
    if filepath == "Not found":
        if verbose:
            print("%s not found in QLP sector %s"%(TIC,sector))
        return
    else:
        print("QLP",filepath)
    
    if savepath == None:
        savepath = "output"
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 

    with open("%s/QLP_%s%s%s_%s_raw.txt"%(savepath,sector,cam,ccd,TIC),"w") as f:
        for i,j in zip(time_raw,signal_raw):
            f.write(",".join([str(i),str(j),"\n"]))

    return










