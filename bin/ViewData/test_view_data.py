"""

This code will try to view the tess output data

"""


from astropy.io import fits
import matplotlib.pyplot as plt


data_path = "../../input/TESS_Data/Orbit9/tess2018224000000-0000000404804908-111-cr_llc.fits"

hdul = fits.open(data_path)

print(hdul.info())

information = hdul[1].data

xlist, ylist = [],[]
for info in information:
    x = info[0]
    y = info[3]
    
    xlist.append(x)
    ylist.append(y)

plt.plot(xlist, ylist)
plt.show()
    
    

