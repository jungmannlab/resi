import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from matplotlib import gridspec



# Data import
path = r'W:\users\reinhardt\z_raw\Resi\211123_dbl-ext-20nm-6sites-R4\workflow_081221\Origami6_distances\Gauss_surface_fit'

R1_filename = os.path.join(path,'R1_ori6.hdf5')
R3_filename = os.path.join(path,'R3_ori6.hdf5')

R1_locs = pd.read_hdf(R1_filename, key = 'locs')
R3_locs = pd.read_hdf(R3_filename, key = 'locs')

Minx = min(R1_locs["x"].min(), R3_locs["x"].min())*130 -5
Miny = max(R1_locs["y"].max(), R3_locs["y"].max())*130 +5

R1_x = R1_locs['x']*130-Minx
R1_y = R1_locs['y']*(-130)+Miny

R3_x = R3_locs['x']*130-Minx
R3_y = R3_locs['y']*(-130)+Miny



# Return a gaussian distribution at an angle alpha from the x-axis
# from astroML for use with curve_fit
def Gaussian2D(xy,*m):
    x, y = xy # (xy = (x,y))
    #print(np.array(m), type(np.array(m)))
    A,x0,y0,varx,vary,rho = np.array(m).flatten()
    X,Y = np.meshgrid(x,y)
    assert rho != 1
    a = 1/(2*(1-rho**2))
    Z = A*np.exp(-a*((X-x0)**2/(varx)+(Y-y0)**2/(vary)-(2*rho/(np.sqrt(varx*vary)))*(X-x0)*(Y-y0)))
    return Z.ravel()

def twelveGaussian2D(xy,*m):
    #print(np.array(m).reshape(12,6))
    p00,p01,p02,p03,p04,p05,p06,p07,p08,p09,p10,p11 = np.array(m).reshape(12,6)
    
    
    g0 = Gaussian2D(xy,p00)
    g1 = Gaussian2D(xy,p01)
    g2 = Gaussian2D(xy,p02)
    g3 = Gaussian2D(xy,p03)
    g4 = Gaussian2D(xy,p04)
    g5 = Gaussian2D(xy,p05)
    g6 = Gaussian2D(xy,p06)
    g7 = Gaussian2D(xy,p07)
    g8 = Gaussian2D(xy,p08)
    g9 = Gaussian2D(xy,p09)
    g10 = Gaussian2D(xy,p10)
    g11 = Gaussian2D(xy,p11)
    return g0+g1+g2+g3+g4+g5+g6+g7+g8+g9+g10+g11


def Hist2D(bins, R_x, R_y):
    #bins = [list(range(50,70,2)),list(range(30,50,2))]
    H, xedges, yedges = np.histogram2d(R_x, R_y, bins=bins)
    H = H.T
    return H, xedges, yedges
    

def InitializeFit(H, xedges, yedges):
    """
    Parameters
    ----------
    H : TYPE
        DESCRIPTION.
    xedges : TYPE
        DESCRIPTION.
    yedges : TYPE
        DESCRIPTION.

    Returns
    -------
    coeff_reshaped : TYPE
        A 12x6 arrays, where each row contains the parameters for one of the twelve Gaussians.
        Each row consists of the follwing parameters: A,x0,y0,varx,vary,rho
    data_fitted : TYPE
        The twelveGaussian2D Function evaluated with the fitting coefficients (the flattend coeff_reshaped array)

    """
    
    # get the centers of the 2d histogram bins
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    
    # Initial guess for peak positions
    maxima = peak_local_max(H, threshold_abs = 10, min_distance=2)
    # plot initial values for peak positions:
    #ax.scatter(xcenters[maxima[:,1]], ycenters[maxima[:,0]], color='r', s=2)
    
    
    # Write initial Initial guesses in a 12x6 array
    p0 = (H.max(),60,37,2,2,0.5,np.pi/4)
    p0 = np.zeros((12,6))
    
    for i in range(len(maxima)):
        p0[i,0] = H[maxima[i,0],maxima[i,1]] # Amplitude
        p0[i,1] = xcenters[maxima[i,1]] # x0
        p0[i,2] = ycenters[maxima[i,0]] # y0
        p0[i,3] = 2 # varx 0
        p0[i,4] = 2 # vary 0
        p0[i,5] = 0.5 # rho 0
        
    
    # Fit 12 Gaussians to the 2d histogram  
    xy = (xcenters, ycenters)
    coeff, covar = curve_fit(twelveGaussian2D,xy,H.ravel(),p0=p0.flatten())
    coeff_reshaped = coeff.reshape(12,6)
    
    # Evaluate the 12 Gaussians function with the fitted coefficients
    data_fitted = twelveGaussian2D(xy,coeff).T
    
    return coeff_reshaped, data_fitted

def plot(path, title, bins, R_x, R_y, H, xedges, yedges, coeff_reshaped, data_fitted): 
    # plot the raw data
    fig = plt.figure(figsize=(15,5))
    plt.suptitle(title)
    spec = gridspec.GridSpec(ncols = 3, nrows = 2, height_ratios = [1, 0.05], figure = fig)
    #fig = plt.figure()
    ax = fig.add_subplot(spec[0,0], title= 'Raw data: Localizations')
    ax.scatter(R_x, R_y, color='r', s=1)
    ax.set_aspect('equal')
    ax.set_xlim((min(bins), max(bins)))
    ax.set_ylim((min(bins), max(bins)))
    
    # plot the histogram of the raw data
    ax2 = fig.add_subplot(spec[0,1], title='Histogram of raw data', aspect='equal')
    XX, YY = np.meshgrid(xedges, yedges)
    cmap = plt.get_cmap('viridis')
    im = ax2.pcolormesh(XX, YY, H, cmap = cmap)
    cax = fig.add_subplot(spec[1,1])
    fig.colorbar(im, cax=cax, orientation="horizontal")
    
    
    # plot 12 Gaussians with the fitted coefficients as well as the determined peak positions
    ax3 = fig.add_subplot(spec[0,2], title='2d Gaussian fit and peak postions', aspect='equal')
    im2 = ax3.pcolormesh(XX, YY, data_fitted.reshape(64,64))
    ax3.scatter(coeff_reshaped[:,1], coeff_reshaped[:,2], color='r', s=5)
    cax = fig.add_subplot(spec[1,2])
    fig.colorbar(im2, cax=cax, orientation="horizontal")
    
    plt.show()
    plt.savefig(os.path.join(path, title + '.png'), transparent=False, bbox_inches='tight')
    plt.savefig(os.path.join(path, title + '.pdf'), transparent=False, bbox_inches='tight')

    
    
# Create a 2d histogram of the raw data
bins = list(range(0,130,2))

H1, xedges1, yedges1 = Hist2D(bins, R1_x, R1_y)

R1_coeff, R1_fitted = InitializeFit(H1, xedges1, yedges1)

plot(path,'R1', bins, R1_x, R1_y, H1, xedges1, yedges1, R1_coeff, R1_fitted)


H3, xedges3, yedges3 = Hist2D(bins, R3_x, R3_y)

R3_coeff, R3_fitted = InitializeFit(H3, xedges3, yedges3)

plot(path,'R3', bins, R1_x, R1_y, H3, xedges3, yedges3, R3_coeff, R3_fitted)

np.savetxt(os.path.join(path, 'R1_FitCoefficients.csv'), R1_coeff, header = 'A,xc,yc,varx,vary,rho')
np.savetxt(os.path.join(path, 'R3_FitCoefficients.csv'), R3_coeff, header = 'A,xc,yc,varx,vary,rho')

R1_x_peaks = R1_coeff[:,1]
R1_y_peaks = R1_coeff[:,2]

R3_x_peaks = R3_coeff[:,1]
R3_y_peaks = R3_coeff[:,2]


'''
distances = np.sqrt((R1_x_peaks-R3_x_peaks)**2+(R1_y_peaks-R3_y_peaks)**2)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, title='Gauss peaks')
ax.scatter(R1_x_peaks, R1_y_peaks, color='r', s = 5)
ax.scatter(R3_x_peaks, R3_y_peaks, color='b', s = 5)
ax.legend()
for i, label in enumerate(distances):
    ax.annotate(str(round(label,2))+'nm', (R1_x_peaks[i], R1_y_peaks[i]))
plt.show()
'''