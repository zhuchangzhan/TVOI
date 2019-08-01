"""

This package will be incorporating different period finding algorithms
that I use for simulations



"""

from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(DIR, '../..'))

import src.main.FFT_Analysis as FFT
import src.main.LC_Analysis as LCA

from PyAstronomy.pyTiming import pyPDM




def fft_transform(x,y,oversampling=8,timestep=1/720):

    N1 = FFT.guess_N(len(x),oversampling,False)
    # TVOI.signal_bin_detrended is reserved for visualization and data folding purpose
    extended_time, extended_signal = LCA.extend_lightcurve(y,N1,0)
    FFT_Result = FFT.compute_fft_general(extended_signal, timestep, N1)  
    freq,sig,amplitude,power = FFT_Result

    return freq,amplitude



def stellingwerf_transform(x,y,peak_frequency):

    target_freq_min = (peak_frequency)*0.999
    target_freq_max = (peak_frequency)*1.001
    
    
    
    print(target_freq_min,target_freq_max)

    S = pyPDM.Scanner(minVal=target_freq_min, maxVal=target_freq_max, dVal=0.00001, mode="frequency")
    
    
    P = pyPDM.PyPDM(x, y)
    
    f,t = P.pdmEquiBin(50, S)
    
    min_index = np.argmin(t)
    
    plt.plot(f,t)
    plt.show()
    
    return f[min_index]






