"""
process injected flares
"""
import os,sys
import glob
import time
import pandas as pd
import matplotlib.pyplot as plt


DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(DIR, '../..'))

import src.main.Lightcurve_io4 as LC_io2
from src.main.General_Catcher import *

def injection_test():
    """
    Run flare catching code over injected lightcurve given by Max
    """
    
    filepaths = glob.glob("injected/*.csv")
    Norm = 0
    counter = 0
    start = time.time()   
    total_time = time.time() -start 
    print("Begin Injected Catching")
    
    savepath = "deploy_injected"
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    
    for count,filepath in enumerate(filepaths):
        
        if count > 84:
            continue
        try:
            TIC_ID = filepath.split("_")[-1].replace(".csv","")
            sector = int(filepath.split("/")[-1].split("_")[0][1:])
            
            df = pd.read_csv(filepath)
            
            times = df["# time"].values
            flux = df["flux"].values
            error = df["flux_err"].values
            
            
            Catcher = SPOC_Catcher_v3(None,None,TIC_ID,sector,Norm)
            Catcher.Load_Lightcurve_Data(True,[],times,flux,error)
            
            TVOI = Catcher.TVOI
            
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


        
        """
        TVOI = LC_io2.SPOC_TVOI(None,None,TIC_ID,sector,Norm)
        TVOI.load_user_input(time,flux,error)
        TVOI.calibrate_lightcurve()
        TVOI.detrend_lightcurve()
        TVOI.bin_lightcurve()
       
        #plt.plot(time,flux)
        plt.plot(TVOI.time_bin,TVOI.signal_bin)
        plt.plot(TVOI.time_bin,TVOI.signal_bin_detrended)
        plt.show()
        """

                

    return

def read_injection():
    
    filepaths1 = glob.glob("injected2/S001*.csv")
    filepaths2 = glob.glob("injected2/S002*.csv")
    
    Norm = 0
    counter = 0
    start = time.time()   
    total_time = time.time() - start 
    
    savepath = "testing_temp2.1"
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    
    listss = glob.glob("injection_params2/*")
    
    
    
    for inject_param in listss:
        print(inject_param)
        
        sector = int(inject_param.split("/")[-1][3])
        
        #if sector != 1:
        #    continue
        
        idf = pd.read_csv(inject_param)
        TIC_IDs    = idf["ID"].values
        tpeaks     = idf["tpeak"].values - 2457000.0
        amplitudes = idf["ampl"].values
        fwhms      = idf["fwhm"].values
        
       
        count = 0
        counter = 0
        
        for TIC,tpeak,ampli,fwhms in zip(TIC_IDs,tpeaks,amplitudes,fwhms):
            #print(TIC,tpeak,ampli,fwhms)
            
            
            if True:
                TIC = str(int(TIC))
                if int(TIC) != 358108509:
                    continue
                print(TIC)
                
                count +=1
                if count%10 == 0:
                    plt.close()
                
                if sector == 1:
                    filepath = [val for val in filepaths1 if TIC in val][0]
                else:
                    filepath = [val for val in filepaths2 if TIC in val][0]
                
        
        
                TIC_ID = filepath.split("_")[-1].replace(".csv","")
                
                
                #sector = int(filepath.split("/")[-1].split("_")[0][1:])
                
                df = pd.read_csv(filepath)
                
                times = df["# time"].values #
                flux = df["flux"].values
                error = df["flux_err"].values
                
                
                Catcher = SPOC_Catcher_v3(None,None,TIC_ID,sector,Norm)
                Catcher.Load_Lightcurve_Data([],[times,flux,error,-1])
                
                TVOI = Catcher.TVOI
                
                if Catcher.TVOI.num_flares > 0:
                    counter +=1
                
                
                
                
                output = Catcher.Create_Flare_Report(savepath,deploy=False,inject = True,
                                        inject_param=[tpeak,ampli,fwhms,sector])
                
                print(count,counter,TIC_ID,output,total_time)
                #else:
                #    print(count,counter,TIC_ID,0,total_time)
                total_time=time.time()-start
            else:
            #except:
                count+=1
                print(count,counter,TIC_ID,"F",total_time)

def read_dropbox_injection():
    
    name = "output_single"
    filepaths1 = glob.glob("/Users/azariven/Dropbox (Personal)/%s/csv/S001*.bz2"%name)
    filepaths2 = glob.glob("/Users/azariven/Dropbox (Personal)/%s/csv/S002*.bz2"%name)

    #filepaths1 = glob.glob("/Users/azariven/Dropbox (Personal)/output/csv/S001*.bz2")
    #filepaths2 = glob.glob("/Users/azariven/Dropbox (Personal)/output/csv/S002*.bz2")
    Norm = 0
    counter = 0
    start = time.time()   
    total_time = time.time() - start 
    
    savepath = "i_boxcar_%s"%name
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    
    listss = glob.glob("/Users/azariven/Dropbox (Personal)/%s/injection_params/*"%name)
    #listss = glob.glob("/Users/azariven/Dropbox (Personal)/output/injection_params/*")
    
    
    for inject_param in listss:
        print(inject_param)
        
        sector = int(inject_param.split("/")[-1][3])
        """
        if "FGK" not in inject_param:
            continue
        
        if sector == 2:
            continue
        """
        
    
        idf = pd.read_csv(inject_param)
        idf = idf[idf["flare_nr"]==0]
        
        TIC_IDs    = idf["ID"].values
        tpeaks     = idf["tpeak"].values - 2457000.0
        amplitudes = idf["ampl"].values
        fwhms      = idf["fwhm"].values
        
        count = 0
        counter = 0
        
        for TIC_ID,tpeak,ampli,fwhms in zip(TIC_IDs,tpeaks,amplitudes,fwhms):
            #print(TIC,tpeak,ampli,fwhms)
            
            
            #try:
            if True:
                TIC_ID = str(TIC_ID)
                #if TIC_ID != "141154638":
                #    continue
                count +=1
                
                
                if count%10 == 0:
                    plt.close()
                if sector == 1:
                    filepath = [val for val in filepaths1 if TIC_ID == val.split("_")[-2]][0]
                else:
                    filepath = [val for val in filepaths2 if TIC_ID == val.split("_")[-2]][0]
                
                
                TESSMAG = float(filepath.split("_")[-1].replace(".csv.bz2",""))
                TIC_ID = filepath.split("_")[-2]#
                
                
                df = pd.read_csv(filepath, compression='bz2', header=0, sep=',', quotechar='"')
                #df = df.rename(columns = {0:"# time",1:"flux",2:"flux_err"})
            
                times = df["# time"].values
                flux  = df["flux"].values
                error = df["flux_err"].values
                
                #print(times)
                
                
                Catcher = SPOC_Catcher_v3(None,None,TIC_ID,sector,Norm)
                Catcher.Load_Lightcurve_Data([],[times,flux,error,TESSMAG])
                
                TVOI = Catcher.TVOI
                
                if Catcher.TVOI.num_flares > 0:
                    counter +=1
                
                
                output = Catcher.Create_Flare_Report(savepath,deploy=True,inject = True,
                                        inject_param=[tpeak,ampli,fwhms,sector])
                
                print(count,counter,TIC_ID,output,total_time)
                #else:
                #    print(count,counter,TIC_ID,0,total_time)
                total_time=time.time()-start
            else:
            #except:
                count+=1
                print(count,counter,TIC_ID,"F",total_time)


def read_returns():
    
    with open("returns.txt","r") as f:
        info = f.read()
        sector = [x.split(" ")[0][3] for x in info.split("\n")][1:]
        bad_tic = [x.split(" ")[1] for x in info.split("\n")][1:]
        

def read_dropbox_injection_fix():


    with open("returns.txt","r") as f:
        info = f.read()
        sector_info = [x.split(" ")[0][3] for x in info.split("\n")][1:]
        bad_tic = [x.split(" ")[1] for x in info.split("\n")][1:]    


    name = "output_outburst"
    filepaths1 = glob.glob("/Users/azariven/Dropbox (Personal)/%s/csv/S001*.bz2"%name)
    filepaths2 = glob.glob("/Users/azariven/Dropbox (Personal)/%s/csv/S002*.bz2"%name)

    #filepaths1 = glob.glob("/Users/azariven/Dropbox (Personal)/output/csv/S001*.bz2")
    #filepaths2 = glob.glob("/Users/azariven/Dropbox (Personal)/output/csv/S002*.bz2")
    Norm = 0
    counter = 0
    start = time.time()   
    total_time = time.time() - start 
    
    savepath = "injection_global_test_%s"%name
    if not os.path.isdir(savepath):
        os.makedirs(savepath) 
    
    listss = glob.glob("/Users/azariven/Dropbox (Personal)/%s/injection_params/*"%name)
    #listss = glob.glob("/Users/azariven/Dropbox (Personal)/output/injection_params/*")
    
    for inject_param in listss:
        #print(inject_param)    

        sector = int(inject_param.split("/")[-1][3])
    
        idf = pd.read_csv(inject_param)
        idf = idf[idf["flare_nr"]==0]
        
        TIC_IDs    = idf["ID"].values
        tpeaks     = idf["tpeak"].values - 2457000.0
        amplitudes = idf["ampl"].values
        fwhms      = idf["fwhm"].values
        
        count = 0
        counter = 0
        
        for TIC_ID,tpeak,ampli,fwhms in zip(TIC_IDs,tpeaks,amplitudes,fwhms):
            #print(TIC,tpeak,ampli,fwhms)
            
            TIC_ID = str(TIC_ID)
            
            for a,b in zip(sector_info,bad_tic):
                if int(a) == sector and b == str(TIC_ID):
                    if sector == 1:
                        filepath = [val for val in filepaths1 if TIC_ID == val.split("_")[-2]][0]
                    else:
                        filepath = [val for val in filepaths2 if TIC_ID == val.split("_")[-2]][0]
                                            
                    
                    #print(filepath)
        
                    TESSMAG = float(filepath.split("_")[-1].replace(".csv.bz2",""))
                    TIC_ID = filepath.split("_")[-2]#
                    
                    
                    df = pd.read_csv(filepath, compression='bz2', header=0, sep=',', quotechar='"')
                    #df = df.rename(columns = {0:"# time",1:"flux",2:"flux_err"})
                
                    times = df["# time"].values
                    flux  = df["flux"].values
                    error = df["flux_err"].values
                    
                    #print(times)
                    
                    
                    Catcher = SPOC_Catcher_v3(None,None,TIC_ID,sector,Norm)
                    Catcher.Load_Lightcurve_Data([],[times,flux,error,TESSMAG])
                    
                    TVOI = Catcher.TVOI
                    
                    if Catcher.TVOI.num_flares > 0:
                        counter +=1
                    
                    output = Catcher.Create_Flare_Report(savepath,deploy=False,inject=True,
                                            inject_param=[tpeak,ampli,fwhms,sector])
                    
                    print(count,counter,TIC_ID,output,total_time)
   
        
        
        
        
        


if __name__ == "__main__":
    #run_all_files()
    #local_run_test()
    #injection_test()
    #read_injection()
    read_dropbox_injection()
    #read_returns()
    #read_dropbox_injection_fix()
