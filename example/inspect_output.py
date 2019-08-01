"""


"""
from collections import Counter
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys
from matplotlib import colors

def inspect_DV_Report():
    
    temperature = []
    period = []
    
    filepath = glob.glob("../../output/Sector9/SPOC_v3/*.png")
    for file in filepath:
        info = file.split("/")[-1].split("_")
        print(info)
        
        temp,p = float(info[4].replace("T","")),float(info[7].replace("P",""))
        
        #if p < 0.05 or p > 1.5:
        #    continue
        
        temperature.append(temp)
        period.append(p)
    
    print(len(temperature))
    
    xy = np.vstack([temperature, period])
    z = gaussian_kde(xy)(xy)
    
    
    plt.scatter(temperature, period, c=z, s=100, edgecolor='')
    #plt.hist2d(temperature, period, (20,20), cmap=plt.cm.jet)
    plt.title("Mstar Temperature vs Period")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Period (day)")
    plt.show()
    
def inspect_output_file(filename,TIC_List = []):
    
    with open(filename) as f:
        data = f.read()
 
        temperatures = []
        freqs = []
        radius = []
        
        for line in data.split("\n"):
            chunk = line.split(",")
            if chunk == [""]:
                continue
            Temp = float(chunk[5].replace("T",""))
            freq = float(chunk[4].replace("Q",""))
            Rstar= float(chunk[6].replace("R",""))
            TIC  = int(chunk[2])
            peaks = float(chunk[-3].replace("P",""))
            
            
            if peaks !=1:
                continue
            
            
            
            
            if TIC in TIC_List:
                continue
            else:
                TIC_List.append(TIC)
            
            """
            if period > 2.4 or period < 0.05:
                continue
            
            
            
            
            
            #if freq < 1:
            #    continue
            
            
            if freq < 0.2:
                continue
            
            if Rstar > 8 or Rstar < 0.01 or Rstar < 1.3:
                continue
            
            
            if Temp > 6500 or Temp < 2700:
                continue
            """
            
            if Temp < 2700 or Temp > 10000:
                continue
            
            if Rstar > 10 or Rstar < 0.01 :
                continue
            
            temperatures.append(Temp)
            freqs.append(freq)
            radius.append(Rstar)
        
        
        
        total_count = [(l,k) for k,l in sorted([(j,i) for i,j in Counter(freqs).items()], reverse=True)]
        
        bad,good = 0,0
        bad_freq = []
        good_freq = []
        for i in total_count:
            
            
            if i[1] > 2:
                bad+=i[1]
                bad_freq.append(i[0])
            else:
                good+=1
                good_freq.append(i[0])
                
        print(good,bad) 
        #print(bad_freq)
        
        
        plot_temp = []
        plot_freq = []
        plot_radi = []
        for fre,tem,rad in zip(freqs,temperatures,radius):
            if fre in good_freq:
                plot_temp.append(tem)
                plot_freq.append(fre)
                plot_radi.append(rad)
        
        return plot_temp,plot_freq,plot_radi,TIC_List
                
            
        #sprint(total_count)
        
        #sys.exit()
      
def collect_multisector():


    output = []
    TIC_ID = []
    for sector in range(9):
        sector +=1
        if sector == 9:
            filename = "../../output/Sector9/SPOC_v3/a.sector9_result.txt"
        else:
            filename = "../../output/Sector%s/a.sector%s_result.txt"%(sector,sector)
        with open(filename) as f:
            data = f.read()
     
            temperatures = []
            freqs = []
            radius = []
            
            for line in data.split("\n"):
                chunk = line.split(",")
                if chunk == [""]:
                    continue
                
                TIC  = chunk[2]
                
                if TIC in TIC_ID:
                    continue
                
                
                hour = chunk[3].replace("hr","")
                freq = chunk[4].replace("Q","")
                Temp = chunk[5].replace("T","")
                Rstar= chunk[6].replace("R","")
                peaks = chunk[7].replace("P","")
                flare = chunk[8].replace("F","")
                
                if freq in freqs:
                    continue
                
                TIC_ID.append(TIC)
                freqs.append(freq)
                
                line = ",".join([TIC,hour,freq,Temp,Rstar,peaks,flare])
                output.append(line)
    with open("total_output_no_dupe.txt","w") as f:
        for i in output:
            f.write(i+"\n")
        
    
                
    

def run_multisector():
    
    

    plot_freq = []
    plot_temp = []
    plot_radi = []
    TIC_List  = []
    
    for sector in range(9):
        sector +=1
        if sector == 9:
            filename = "../../output/Sector9/SPOC_v3/a.sector9_result.txt"
        else:
            filename = "../../output/Sector%s/a.sector%s_result.txt"%(sector,sector)
        a,b,c,TIC_List = inspect_output_file(filename,TIC_List)
        plot_temp = np.concatenate([plot_temp,a])
        plot_freq = np.concatenate([plot_freq,b])
        plot_radi = np.concatenate([plot_radi,c])
    
    print(len(TIC_List))



    xy = np.vstack([plot_temp,1/np.array(plot_freq)])
    xy = np.vstack([plot_temp, plot_freq])
    z = gaussian_kde(xy)(xy)
    
    
    """
    from mpl_toolkits.mplot3d import axes3d
    fig, ax = plt.subplots()
    ax = fig.gca(projection='3d')
    ax.scatter(plot_temp,1/np.array(plot_freq),plot_radi,c=plot_radi)
    """
    plt.scatter(plot_temp, 1/np.array(plot_freq), c=z, edgecolor='')
    #plt.hist2d(plot_temp, 1/np.array(plot_freq), (100,100), cmap=plt.cm.jet)#, norm=colors.LogNorm())
    #plt.plot(temperatures, periods,".")
    #plt.yscale("log")
    plt.title("Temperature vs Period (Sector 1-9)")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Period (Days)")
    
    plt.tight_layout()
    #plt.zlabel("Radius (Rsun)")  
      
    #cbar = plt.colorbar()
    
    #cbar.ax.set_ylabel('Counts')

    plt.show()

if __name__ == "__main__":
    #inspect_output_file()
    run_multisector()
    #collect_multisector()
    
    
    
    
    
    
    
    
    
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate fake data
x = np.random.normal(size=1000)
y = x * 3 + np.random.normal(size=1000)

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=100, edgecolor='')
plt.show()

"""