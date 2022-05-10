#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:12:52 2022

@author: masullo
"""

import os
from os.path import dirname as up

cwd = os.getcwd()
wdir = up(cwd)
os.chdir(wdir)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import tools


plt.close('all')

pxsize = 130 # nm
simulate = True

if simulate:

    # origami sites (built manually) 
    
    means_ch1 = np.array([[-20, 0], [0, 0], [20, 0]])
    
    means_ch2 = means_ch1.copy()
    
    dx = 0.0 # in nm
    
    dx_array = np.array([[dx, 0], [dx, 0], [dx, 0]])
    
    means_ch2 = means_ch2 + dx_array
    sites = np.concatenate((means_ch1, means_ch2), axis=0)
    
    simulated_data_fname = os.getcwd() + '/simulated/simulated_data_2.hdf5' 
    tools.simulate_data(simulated_data_fname, sites, locs_per_site=200, 
                        σ_dnapaint=2.0) # creates simulated data file

# resample for different K
    
files = ['simulated/simulated_data_2.hdf5']

# K_array = np.array([1, 5, 10, 20, 30, 40]) # number of localizations per subset

K_array = np.array([1])

# fig0, ax0 = plt.subplots(2, 3, figsize=(20, 16)) # size matches len(K_array)

fig, ax = plt.subplots(1, 1, figsize=(20, 16)) # size matches len(K_array)

iterables = K_array

for k, K in enumerate(K_array):

    resi_data, pre_resi_data = tools.get_resi_locs(files, K)

    all_locs_x = np.array(resi_data['0']['x']) # simulation already in nm
    all_locs_y = np.array(resi_data['0']['y']) # simulation already in nm
    
    binsmax = all_locs_x.max() + 10
    binsmin = all_locs_x.min() - 10

    bins = np.arange(binsmin, binsmax, 2)
    
    # ax.hist2d(all_locs_x, all_locs_y, bins=bins, cmap='hot')
    ax.scatter(all_locs_x, all_locs_y, color='#1D70A2')
    ax.title.set_text('K = {}'.format(K))
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_aspect('equal')
    
    counts, bins, _ = np.histogram2d(all_locs_x, all_locs_y, bins=bins)
    
    blue_colormap = np.zeros((256,3))
    blue_colormap[:,2] = np.reshape(np.linspace(0,1,num=256), (256,))
    
    M_color = np.zeros((counts.shape[0],counts.shape[1],3))
    color = 'yellow'
    
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if color == 'red':
                M_color[j,i,0] = counts[i,j]/np.max(counts)
            elif color == 'green':
                M_color[j,i,1] = counts[i,j]/np.max(counts)
            elif color == 'blue':
                M_color[j,i,2] = counts[i,j]/np.max(counts)
                M_color[j,i,1] = 0.6 * counts[i,j]/np.max(counts)
            elif color == 'yellow':
                M_color[j,i,0] = counts[i,j]/np.max(counts)
                M_color[j,i,1] = counts[i,j]/np.max(counts)
  
plt.figure()
plt.imshow(M_color)

counts1D = np.sum(counts, axis=1)

fig1, ax1 = plt.subplots()

ax1.bar(bins[:-1], counts1D, width=1, edgecolor='k', color='#27AAE1',
        alpha=0.6, linewidth=0.1)

#Define the Gaussian function

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

popt, pcov = curve_fit(multi_gauss, bins[:-1], counts1D, p0=init_guess)

ax1.plot(np.linspace(bins[0], bins[-2], 1000), 
         multi_gauss(np.linspace(bins[0], bins[-2], 1000), *popt),
         color='#27AAE1', linewidth=2)
    

    

    
