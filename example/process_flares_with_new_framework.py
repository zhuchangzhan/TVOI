"""

I need to be able to catch flares using the latest framework

What I need to do:
    1. initiate with what I'd done before for variable stars
    2. Replicate what I did for flares

So there may be a flare catcher follow by a driver file?

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




def local_run_test():
    
    sector = 2
    Norm = 0
    
    #filepaths = glob.glob("../../input/Sector9/*.gz")
    filepaths = return_all_sector_datapath(sector)
   
    #filepaths = glob.glob("../../input/Sector%s/test/*.gz"%sector)
    savepath = "deploy_sector_%s"%sector
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    
    counter = 0
    start = time.time()   
    total_time = time.time() -start 
    print("Begin Sector %s Catching"%sector)
    for count,filename in enumerate(filepaths):
        
        #if count > 1687 and count < 8792:
        #    continue
        
        try:
        
            TIC_ID = int(filename.split("/")[-1].split("-")[2])
            Catcher = SPOC_Catcher_v3(filename,savepath,TIC_ID,sector,Norm)
            Catcher.Load_Lightcurve_Data()
            
            if Catcher.TVOI.num_flares > 0:
                output = Catcher.Create_Flare_Report(savepath,deploy=True)
                
                if counter%20 == 0:
                    plt.close()
                
                counter +=1
                print(count,counter,TIC_ID,output,total_time)
            else:
                print(count,counter,TIC_ID,0,total_time)
        except:
            print(count,counter,TIC_ID,"F",total_time)
            with open(os.path.join(savepath,"a.sector%s_result.txt"%sector),"a") as outputfile:
                outputfile.write(",".join([str(count),str(counter),str(TIC_ID),"F",str(total_time),"\n"]))
        total_time=time.time()-start

def injection_test():
    """
    Run flare catching code over injected lightcurve given by Max
    """
    
    filepaths = glob.glob("injected/*.csv")
    print(len(filepaths))
    for filepath in filepaths:
        print(filepath)
    
    
    
    
    
    return




if __name__ == "__main__":
    #run_all_files()
    #local_run_test()
    injection_test()





















