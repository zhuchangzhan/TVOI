import os
import glob
import numpy as np


Known_Planet = {"HATS-3":336732616,
                "HATS-13":289793076,
                "HATS-14":29344935,
                "HATS-30":281459670,
                "HATS-34":355703913,
                "HATS-46":281541555,
                "HATS-68":322307342,
                "WASP-46":231663901,
                "WASP-62":149603524,
                "WASP-73":231670397,
                "WASP-91":238176110,
                "WASP-94":92352620,
                #"WASP-94b":92352621,
                "WASP-95":144065872,
                "WASP-100":38846515,
                "WASP-111":25375553,
                "WASP-119":388104525,
                "WASP-124":97409519}
Known_Planet_TIC_ID = {v: k for k, v in Known_Planet.items()}



def return_known_planet_datapaths(deploy=False):
    
    #print(Known_Planet)
    if deploy:
        return []
    else:
        filepaths = []
        datapath = "../../input/TESS_Data/TOI/"

        for planet in Known_Planet:
            TIC_ID = Known_Planet[planet]
            filepaths.append(glob.glob("%s/*%s*.fits.gz"%(datapath,TIC_ID))[0])
    
        return filepaths

def return_all_planet_datapaths(deploy=False):

    
    if deploy:
        return []
    else:
        filepaths = []
        datapath = "../../input/TESS_Data/TOI/"

        for planet in Known_Planet:
            TIC_ID = Known_Planet[planet]
            filepaths.append(glob.glob("%s/*%s*.fits.gz"%(datapath,TIC_ID))[0])
    
        return filepaths


def return_single_target_sector_available(target):
    pass
  
def return_single_target_datapath(sector,TIC,source="SPOC"):
 
    if sector == -1:
        return ""
    
    if source == "SPOC":
        if sector == 1:
            datapath = "/pdo/spoc-data/sector-01/sector_early_look/light-curve/"
        elif sector >= 2 and sector < 10:
            datapath = "/pdo/spoc-data/sector-0%s/light-curve/"%sector
        elif sector >= 10 and sector < 26:
            datapath = "/pdo/spoc-data/sector-%s/light-curve/"%sector
        else:
            return ""   
        
        try:
            filepath = glob.glob("%s*%s*.fits.gz"%(datapath,TIC))[0]
        except:
            filepath = []
        
        if filepath == [] or filepath == "":
            return "Not found"
        else:
            return filepath
    
    else:
        for cam in list(np.arange(4)+1):
            for ccd in list(np.arange(4)+1):
                datapath = "/pdo/qlp-data/sector-%s/ffi/cam%s/ccd%s/LC/%s.h5"%(sector,cam,ccd,TIC)
                if os.path.isfile(datapath):
                    return datapath,cam,ccd
        return "Not found",0,0

    

def return_interested_target_datapath(sector=-1,target=[]):
    """
    This function is not well optimized when target list is very long
    
    """

    if sector == -1:
        return []
    
    if sector == 1:
        datapath = "/pdo/spoc-data/sector-01/sector_early_look/light-curve/"
    elif sector >= 2 and sector < 10:
        datapath = "/pdo/spoc-data/sector-0%s/light-curve/"%sector
    elif sector >= 10 and sector < 26:
        datapath = "/pdo/spoc-data/sector-%s/light-curve/"%sector
    else:
        return []

    filepaths = []
    for TIC in target:
        try:
            filepath = glob.glob("%s*%s*.fits.gz"%(datapath,TIC))[0]
            filepaths.append(filepath)
            #print(filepath)
        except:
            print(datapath, "not found")
            
    
    
    return filepaths                  

def return_all_sector_datapath(sector=-1):
    """
    """
    output = "input/sector%s_filepath.txt"%sector
    
    if not os.path.isdir("input"):
        os.makedirs("input")
    
    if os.path.isfile(output):
        filepaths = np.genfromtxt(output,dtype="str")
        if len(filepaths) !=0:
            print("Sector %s Filepath Loaded"%sector)
            return filepaths
        
    
    if sector == -1:
        return []
    
    if sector == 1:
        datapath = "/pdo/spoc-data/sector-01/sector_early_look/light-curve/"

    elif sector >= 2 and sector < 10:
        datapath = "/pdo/spoc-data/sector-0%s/light-curve/"%sector
    
    elif sector >= 10 and sector < 26:
        datapath = "/pdo/spoc-data/sector-%s/light-curve/"%sector
    
    else:
        return []
    
    

    filepaths = glob.glob("%s*.fits.gz"%datapath)
    np.savetxt(output,filepaths,fmt="%s")
    
    return filepaths

    
    
    
    

