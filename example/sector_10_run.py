"""


setting up for running sector 8. changes need to be applied to the new code

1. 


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

from scipy.signal import find_peaks

DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(DIR, '../..'))

from src.main.General_Catcher import *
import src.util.configurable as config

user_input = config.Configuration("../../input/configs/user_input_test.cfg")



def run_all_files():
    
    name = "General_period"
    Norm = 0 # normalization factor
    sector = 10
    
    filepaths = return_all_sector_datapath(sector)
    #filepaths = glob.glob("../../input/Sector10/test/*.gz")
    savepath = "output_sec%s_%s"%(sector,name)
    
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    
    savepath1 = "output_sec%s_%s/DV"%(sector,name)
    savepath2 = "output_sec%s_%s/Data"%(sector,name)
    savepath3 = "output_sec%s_%s/FFT"%(sector,name)
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    if not os.path.isdir(savepath1):
        os.makedirs(savepath1) 
    if not os.path.isdir(savepath2):
        os.makedirs(savepath2) 
    if not os.path.isdir(savepath3):
        os.makedirs(savepath3)    
    
    total_time = 0
    target_catch = 0 
    print("Begin Sector %s Catching"%sector)
    for count,filename in enumerate(filepaths):
        
        TIC_ID = int(filename.split("/")[-1].split("-")[2])
        
        if count < 19820:
            continue
    
        start = time.time()   
        try:
        
            cutoffs = []
            #cutoffs["Temp"] = 5000
        
            Catcher = SPOC_Catcher_v3(filename,savepath1,TIC_ID,sector,Norm)
            Catcher.Load_Lightcurve_Data(cutoffs)
            output = Catcher.Conduct_FFT_Analysis()
            
            if output == 0:
                print(count+1,target_catch,TIC_ID,0,0,"D:%.2f"%(time.time()-start),"T:%.2f"%total_time)
            elif output == 1:
                target_catch+=1
                
                if Catcher.predict_frequency > 1.0:
                    Catcher.Generate_1Pager(deploy=True) 
                
                TVOI = Catcher.TVOI
                
                try:
                    Temp  = float(TVOI.object["TEFF"])
                except:
                    Temp = 0
                
                try:
                    Rstar = float(TVOI.object["RADIUS"])
                except:
                    Rstar = 0
                
                
                
                print(count+1,target_catch,TIC_ID,
                      "%.3fhr"%(24/Catcher.predict_frequency),"T:%s"%(Temp),
                      Catcher.num_peaks,str(TVOI.num_flares),"D:%.2f"%(time.time()-start),"F:%.2f"%total_time)
                
                with open(os.path.join(savepath,"a.sector%s_result.txt"%sector),"a") as outputfile:
                    outputfile.write(",".join([str(count+1),
                                               str(target_catch),
                                               str(TIC_ID),
                                               "%.6fhr"%(24/Catcher.predict_frequency),
                                               "Q%.9f"%(Catcher.predict_frequency),
                                               "T%s"%(str(Temp)),
                                               "R%s"%(str(Rstar)),
                                               "P%s"%(str(Catcher.num_peaks)),
                                               "F%s"%(str(TVOI.num_flares)),
                                               "\n"]))
                
                """
                #print(Catcher.should_generate,len(Catcher.sorted_peaks),1/Catcher.predict_frequency)
                with open("%s/SPOC_%s_%s_fold.txt"%(savepath2,sector,TIC_ID),"w") as f:
                    for i,j in zip(Catcher.fold_lc_centers,Catcher.fold_lc_bin_means):
                        f.write(",".join([str(i),str(j),"\n"]))
                """
                if Catcher.num_peaks > 4:
                    with open("%s/SPOC_%s_%s_FFT.txt"%(savepath3,sector,TIC_ID),"w") as f:
                        for i,j in zip(Catcher.freq[Catcher.peak_index],Catcher.amplitude[Catcher.peak_index]):
                            f.write(",".join([str(i),str(j),"\n"]))
                    
            
        
        except:
            print(count+1,target_catch,TIC_ID,"F","F","D:%.2f"%(time.time()-start),"T:%.2f"%total_time)
            
            with open(os.path.join(savepath,"a.sector%s_error.txt"%sector),"a") as errorfile:
                
                errorfile.write(",".join([str(count+1),str(target_catch),str(TIC_ID),
                                          "F","F","D:%.2f"%(time.time()-start),"T:%.2f"%total_time,"\n"]))
        
        total_time+=time.time()-start

def local_run_test():
    
    sector = 10
    name = "Test"
    Norm = 0
    
    #filepaths = glob.glob("../../input/Sector9/*.gz")
    filepaths = glob.glob("../../input/Sector10/test/*.gz")
    savepath = "output_sec%s_%s"%(sector,name)
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
        
    for count,filename in enumerate(filepaths):
        #print(filename)
        
        TIC_ID = int(filename.split("/")[-1].split("-")[2])
        Catcher = SPOC_Catcher_v3(filename,savepath,TIC_ID,sector,Norm)
        Catcher.Load_Lightcurve_Data()
        output = Catcher.Conduct_FFT_Analysis()
        """
        plt.plot(Catcher.TVOI.time_calibrated,Catcher.TVOI.signal_calibrated)
        plt.plot(Catcher.TVOI.time_bin*Catcher.TVOI.bin2day,Catcher.TVOI.signal_bin)
        plt.show()
        """
        if output !=0:
            print(TIC_ID)
            Catcher.Generate_1Pager(deploy=True,forced=False)
        
        #plt.plot(Catcher.TVOI.time_raw, Catcher.TVOI.signal_raw,".")
        #plt.show()  
        
         
         
if __name__ == "__main__":
    run_all_files()
    #local_run_test()
    
    
    
    