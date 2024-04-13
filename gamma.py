from cProfile import label
from cmath import tau
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc("text",usetex=True)
import scipy as sp
import scipy.interpolate
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from matplotlib.ticker import MultipleLocator
import matplotlib.image as mpimg
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
import time

# Import the python package for cross-section
from pack_gato import *

fs=22

# Define a custom tick formatter to display 10^x
def log_tick_formatter(x, pos):
    return r'$10^{%d}$' % int(x)

# Record the starting time
start_time = time.time()

# Flux from Eq. 26 of Kafexhiu et al. 2014
def func_Jp(Tp):

    p=np.sqrt((Tp+mp)**2-mp**2)

    return p**-2*np.exp(-p/1.0e13)

# Gamma-ray plot compared to Fig. 15 of Kafexhiu et al. 2014
def plot_gamma_K14():

    # Arrays for cosmic-ray energy (Tp in eV) and gamma-ray energy (Eg in eV)
    Tp=np.logspace(8.0,15.0,5001)
    dTp=Tp[1:-1]-Tp[0:-2]
    Tp=Tp[0:-2]
    Eg=np.logspace(8.0,13.0,100)

    # Compute nuclear enhancement factor and gamma-ray differential cross-section
    eps_nucl=func_enhancement(Tp) 
    d_sigma_g=func_d_sigma_g(Tp,Eg)
    
    # Compute the gamma-ray flux 
    phi_K14=np.sum(4.0*np.pi*(dTp*func_Jp(Tp)*eps_nucl)[:,np.newaxis]*d_sigma_g, axis=0)

    # Plot results to compare with Kafexhiu's results (note that we multiply the flux with a constant factor since normalization is not known from Kafexhiu et al. 2014)
    fig=plt.figure(figsize=(10, 8))
    ax=plt.subplot(111)
    ax.plot(np.log10(Eg*1.0e-9),np.log10(Eg**2*phi_K14*7.85e25),'r:',linewidth=5.0, label=r'{\rm Minh}')

    # Read the image for data    
    img = mpimg.imread("Kafexhiu_2014.png")
    img_array = np.mean(np.array(img), axis=2)

    xmin=-2.0
    xmax=np.log10(3.0e4)
    ymin=-3.0
    ymax=1.0
    ax.imshow(img_array, cmap ='gray', extent =[xmin, xmax, ymin, ymax], interpolation ='nearest', origin ='upper') 
    ax.set_xticks(np.arange(xmin,xmax,1))
    ax.set_yticks(np.arange(ymin,ymax+1,1))

    # ax.set_ylim(-1,1)

    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    ax.legend()
    ax.set_aspect(1.2)
    ax.set_xlabel(r'$E_\gamma\, {\rm (GeV)}$',fontsize=fs)
    ax.set_ylabel(r'$E_\gamma^2\phi(E_\gamma) \, ({\rm GeV\, cm^{-3}\, s^{-1}})$',fontsize=fs)
    for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_ax.set_fontsize(fs)
    ax.legend(loc='lower left', prop={"size":fs})
    ax.grid(linestyle='--')

    plt.savefig('fg_gamma_K14.png')

plot_gamma_K14()

# Record the ending time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time, "seconds")