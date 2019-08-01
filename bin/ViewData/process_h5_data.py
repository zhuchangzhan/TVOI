"""

learning code to convert h5 data into txt data

"""


import h5py
import numpy as np
import matplotlib.pyplot as plt


"""
CatalogueMagnitudes
LightCurve
    AperturePhotometry
        Aperture_000
            COSMagnitude
            KSPMagnitude
            QualityFlag
            RawMagnitude
            RawMagnitudeError
        Aperture_001
        Aperture_002
        Aperture_003
        Aperture_004
    BJD
    Background
        Error
        Value
    Cadence
    X
    Y


"""
import os





def look():
    
    
    
    
    name = "141768070_orbit9"
    
    path = "../../input/TESS_Data/Lightcurve/lc/%s.h5"%name
    
    f = h5py.File(path, 'r')
    
    for i in f.keys():
        print(i)
    
    magnitude = f["CatalogueMagnitudes"]
    
    lightcurve = f["LightCurve"]
    
    time = lightcurve["BJD"]
    
    for i in lightcurve:
        print(i)
    
    
    file = open("../../input/TESS_Data/Lightcurve/lc_output/%s.txt"%name,"r")
    xlist, ylist = [],[]
    for i,rawflux in enumerate(lightcurve["AperturePhotometry"]["Aperture_002"]["RawMagnitude"]):
    #for i,rawflux in enumerate(lightcurve["Background"]["Error"]):
        file.write("%s %s\n")%(time[i],rawflux)
        xlist.append(time[i])
        ylist.append(rawflux)
    
    file.close()
    plt.plot(xlist,ylist)
    
    
    plt.show()
    

def group_convert():
    
    data_path = "../../input/TESS_Data/Lightcurve/lc/"
    for name in os.listdir(data_path):
        print(name)
    
        name = name.split(".")[0]
        #name = "141768070_orbit9"
        
        path = data_path+"%s.h5"%name
        
        f = h5py.File(path, 'r')
        
        magnitude = f["CatalogueMagnitudes"]
        
        lightcurve = f["LightCurve"]
        
        time = lightcurve["BJD"]
        
        file = open("../../input/TESS_Data/Lightcurve/lc_output/%s.txt"%name,"w")
        xlist, ylist = [],[]
        for i,rawflux in enumerate(lightcurve["AperturePhotometry"]["Aperture_002"]["RawMagnitude"]):
        #for i,rawflux in enumerate(lightcurve["Background"]["Error"]):
            file.write("%s %s\n"%(str(time[i]),str(rawflux)))
            xlist.append(time[i])
            ylist.append(rawflux)
        
        file.close()

if __name__ == "__main__":
    group_convert()
    #look()