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

# pxsize = 130 # nm
simulate = True
σ_dnapaint = 3.0


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
    
    
if simulate:
    
    locs_per_site = 1200
    simulated_data_fname = os.getcwd() + '/simulated/simulated_data' + str(locs_per_site) +'.hdf5' 
    tools.simulate_data(simulated_data_fname, sites, locs_per_site=locs_per_site, 
                        σ_dnapaint=3.0) # creates simulated data file
    
    
### plot ground truth ###

fig0, ax0 = plt.subplots()

ax0.scatter(sites[:, 0], sites[:, 1], marker='X', linewidths=0.5)
ax0.set_xlabel('x (nm)')
ax0.set_ylabel('y (nm)')
ax0.set_aspect('equal')
ax0.set_xlim(-20, 20)
ax0.set_ylim(-30, 30)

### resample for different K ###
    
files = ['simulated/simulated_data1200.hdf5']

K_array = np.array([1, 10, 30, 50, 80, 100]) # number of localizations per subset

fig3, ax3 = plt.subplots(2, 3, figsize=(20, 16)) # size matches len(K_array)

iterables = [ax3.reshape(-1), K_array]

channel_keys = ['0']
minsize = 5 # minimum number of resi localizations needed to consider the statistics
σ_resi_mean_array = np.zeros(len(K_array))
σ_resi_std_array = np.zeros(len(K_array))
counter = np.zeros(len(K_array)) # counter to keep track of how many σ will be considered for each K

for k, (ax, K) in enumerate(zip(*iterables)):

    resi_data, pre_resi_data = tools.get_resi_locs(files, K)

    all_locs_x = np.array(resi_data['0']['x']) # simulation already in nm
    all_locs_y = np.array(resi_data['0']['y']) # simulation already in nm
    
    # binsmax = all_locs_x.max() + 10
    # binsmin = all_locs_x.min() - 10
    
    binsmax = 40
    binsmin = -40

    bins = np.arange(binsmin, binsmax, 0.3)
    
    ax.hist2d(all_locs_x, all_locs_y, bins=bins, cmap='hot')
    ax.title.set_text('K = {}'.format(K))
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_aspect('equal')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-30, 30)
    
    σ_resi_list = []
    
    if K == 1:
        
        # fig1, ax1 = plt.subplots()
        # ax1.scatter(all_locs_x, all_locs_y)
        # ax1.set_xlabel('x (nm)')
        # ax1.set_ylabel('y (nm)')
        # ax1.set_aspect('equal')
        # ax1.set_xlim(-20, 20)
        # ax1.set_ylim(-30, 30)
        
        cluster_size_array = [] # keep track of the size of the clusters
        
    for key in channel_keys:
        
        cluster_id_set = list(set(list(resi_data[key]['cluster_id'])))

        # for i in range(nclusters):
        for _ , cluster_id in enumerate(cluster_id_set):
        
            # get each cluster localizations
            cluster = resi_data[key].loc[resi_data[key]['cluster_id'] == cluster_id]
            cluster_size = cluster.shape[0]
            
            if K == 1:
                cluster_size_array.append(cluster_size)
            
            if cluster_size > minsize:
                
                # calculate covariance matrix (WARNING: no "px_size" factor in simulations)
                cov = np.cov(cluster['x'], cluster['y'])

                # calculate 1D σ
                σ_resi = np.sqrt(1/2 * np.trace(cov))
                
                # save σ_resi into the list
                σ_resi_list.append(σ_resi)
                
                counter[k] += 1
    
    # calculate mean, std and save σ_mean and σ_std for every K
    σ_resi_mean = np.array(σ_resi_list).mean() 
    σ_resi_std = np.array(σ_resi_list).std()
     
    σ_resi_mean_array[k] = σ_resi_mean
    σ_resi_std_array[k] = σ_resi_std

print('Number of valid clusters (size > 5 locs) for each K are: ', counter)

cluster_size_array = np.array(cluster_size_array)

print('Average cluster size: ', cluster_size_array.mean())

# plot results
     
fig, ax = plt.subplots()

ax.errorbar(K_array, σ_resi_mean_array, yerr=σ_resi_std_array, fmt='ko', 
            capsize=5, label='Measured data', zorder=2)

ax.plot(np.arange(1, 300, 0.2), σ_dnapaint/np.arange(1, 300, 0.2)**(1/2), 
        color='#266DD3', label='$σ_{dna-paint} / K^{1/2}$')

ax.set_xlabel('K')
ax.set_ylabel('$σ_{resi}$ (nm)')

ax.set_xlim(-5, 125)
ax.set_ylim(0.1, 3.2)

ax.legend()


### plot resampling ###


    
