#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:12:52 2022

@author: masullo
"""

import os
from os.path import dirname as up

cwd = os.getcwd()
wdir = up(up(cwd))
os.chdir(wdir)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import tools


plt.close('all')
plt.style.use('dark_background')

pxsize = 130 # nm
simulate = True

ticks = True # change to True to show ticks and labels


if simulate:

    # origami sites (built manually) 
    
    means_ch1 = np.array([[-21, 0], [-1, 0], [19, 0]])
    means_ch2 = means_ch1.copy()
    
    dx = 2.0 # in nm
    dx_array = np.array([[dx, 0], [dx, 0], [dx, 0]])
    
    means_ch2 = means_ch2 + dx_array
    sites = np.concatenate((means_ch1, means_ch2), axis=0)
    
    simulated_data_fname_1_2 = os.getcwd() + '/simulated/simulated_data_ch1_ch2.hdf5' 
    tools.simulate_data(simulated_data_fname_1_2, sites, locs_per_site=200, 
                        σ_dnapaint=3.0) # creates simulated data file
    
    simulated_data_fname_1 = os.getcwd() + '/simulated/simulated_data_ch1.hdf5' 
    tools.simulate_data(simulated_data_fname_1, means_ch1, locs_per_site=200, 
                        σ_dnapaint=3.0) # creates simulated data file
    
    simulated_data_fname_2 = os.getcwd() + '/simulated/simulated_data_ch2.hdf5' 
    tools.simulate_data(simulated_data_fname_2, means_ch2, locs_per_site=200, 
                        σ_dnapaint=3.0) # creates simulated data file

# resample for different K
    
color_scatter = '#D90429'
color_hist = '#D90429'
color_fit = '#D90429'

file_1_2 = ['simulated/simulated_data_ch1_ch2.hdf5']

K = 1 # note that RESI with K = 1 is equivalent to DNA-PAINT

fig, ax = plt.subplots(4, 1, figsize=(8, 20)) # size matches len(K_array)

resi_data_ch1_ch2, pre_resi_data_ch1_ch2 = tools.get_resi_locs(file_1_2, K) 

all_locs_x_ch1_ch2 = np.array(resi_data_ch1_ch2['0']['x']) # simulation already in nm
all_locs_y_ch1_ch2 = np.array(resi_data_ch1_ch2['0']['y'])
    
ax[0].scatter(all_locs_x_ch1_ch2, all_locs_y_ch1_ch2, color=color_scatter, s=7.5)
# ax[1].scatter(all_locs_x_ch2, all_locs_y_ch2, color='#9EE09E')

ax[0].set_aspect('equal')

ax[0].set_xlim(-40, 40)
# ax[1].set_xlim(-30, 30)

ax[0].set_ylim(-15, 15)
# ax[1].set_ylim(-6, 6)

bins = np.arange(-30, 30, 1)
    
counts_ch1_ch2, bins_ch1_ch2, _ = np.histogram2d(all_locs_x_ch1_ch2, 
                                                 all_locs_y_ch1_ch2, 
                                                 bins=bins, density=True)

counts1D_ch1_ch2 = np.sum(counts_ch1_ch2, axis=1)

bins_centers_ch1_ch2 = (bins_ch1_ch2[:-1] + bins_ch1_ch2[1:])/2
ax[1].bar(bins_centers_ch1_ch2, counts1D_ch1_ch2, width=1, edgecolor='k', 
          color=color_hist, alpha=0.5, linewidth=0.1)
             
# Define the Gaussian function

def gaussian(x, A, x0, σ):
    
    return A*np.exp(-(x-x0)**2/(2*σ**2))

def multi_gauss(x, *pars):
    
    
    offset = pars[-1]
    offset = 0 # force offset to be 0
    
    g1 = gaussian(x, pars[0], pars[1], pars[2])
    g2 = gaussian(x, pars[3], pars[4], pars[5])
    g3 = gaussian(x, pars[6], pars[7], pars[8])
    
    return g1 + g2 + g3 + offset

init_guess = np.array([10, -20, 2, 10, 0, 2, 10, 20, 2])  

popt_ch1_ch2, pcov_ch1_ch2 = curve_fit(multi_gauss, bins_centers_ch1_ch2, 
                                       counts1D_ch1_ch2, p0=init_guess)

ax[1].plot(np.linspace(-40, 40, 2000), 
            multi_gauss(np.linspace(-40, 40, 2000), *popt_ch1_ch2),
            color=color_fit, linewidth=3)
    
ax[1].set_xlim(-40, 40)

if ticks == False:
    
    ax[0].tick_params(left = False, right = False , labelleft = False ,
                      labelbottom = False, bottom = False)
    ax[1].tick_params(left = False, right = False , labelleft = False ,
                      labelbottom = False, bottom = False)

plt.savefig("figure1_c_high_res.png", dpi=1200)

