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
import tools

plt.close('all')

pxsize = 130 # nm
simulate = False

if simulate:

    # origami sites (built manually) 
    
    means_ch1 = np.array([[-30, -7.5], [-30, 7.5], [30, 0], 
                          [40, 0], [40, 20], [40, -15],
                          [-10, 20], [-10, 0], [-10, -20],
                          [10, 20], [10, 0], [10, -20]])
    
    means_ch2 = means_ch1.copy()
    
    dx = 2.0 # in nm
    
    dx_array = np.array([[0, 0], [0, 0], [0, 0], 
                         [0, 0], [0, 0], [0, 0],
                         [dx, 0], [dx, 0], [dx, 0],
                         [dx, 0], [dx, 0], [dx, 0]])
    
    means_ch2 = means_ch2 + dx_array
    sites = np.concatenate((means_ch1, means_ch2), axis=0)
    
    simulated_data_fname = os.getcwd() + '/simulated/simulated_data.hdf5' 
    tools.simulate_data(simulated_data_fname, sites, locs_per_site=8000, 
                        σ_dnapaint=3.0) # creates simulated data file

# resample for different K
    
files = ['simulated/simulated_data.hdf5']

K_array = np.array([1, 5, 10, 20, 30, 40]) # number of localizations per subset

fig0, ax0 = plt.subplots(2, 3, figsize=(20, 16)) # size matches len(K_array)

iterables = [ax0.reshape(-1), K_array]

for k, (ax, K) in enumerate(zip(*iterables)):

    resi_data, pre_resi_data = tools.get_resi_locs(files, K)

    all_locs_x = np.array(resi_data['0']['x']) # simulation already in nm
    all_locs_y = np.array(resi_data['0']['y']) # simulation already in nm
    
    binsmax = all_locs_x.max() + 10
    binsmin = all_locs_x.min() - 10

    bins = np.arange(binsmin, binsmax, 0.3)
    
    ax.hist2d(all_locs_x, all_locs_y, bins=bins, cmap='hot')
    ax.title.set_text('K = {}'.format(K))
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_aspect('equal')
    

    
