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
import tools

plt.close('all')

# files = ['test_files/full_aligned_clusterd_data_R1_apicked.hdf5', 
#          'test_files/full_aligned_clusterd_data_R3_apicked.hdf5']

# channel_keys = ['0', '1']

# files = ['test_files/figure3_k_resampling/nodel_R1_etc_ClusterD4_50_picked_multi_picked.hdf5',
#          'test_files/figure3_k_resampling/nodel_R3_etc_aligned_ClusterD4_50_picked_multi_picked.hdf5']

files = ['test_files/figure3_k_resampling/R1_ClusterD4_50_merged.hdf5', 
         'test_files/figure3_k_resampling/R3_ClusterD4_50_merged.hdf5']

channel_keys = ['0', '1']

pxsize = 130 # nm
σ_dnapaint = 1.78 # nm


K_array = np.array([1, 2, 3, 5, 10, 20, 30, 40, 60, 80, 100]) # number of localizations per subset
minsize = 5 # minimum number of resi localizations needed to consider the statistics
σ_resi_mean_array = np.zeros(len(K_array))
σ_resi_std_array = np.zeros(len(K_array))
counter = np.zeros(len(K_array)) # counter to keep track of how many σ will be considered for each K

# iterate over the array of different K values
for k, K in enumerate(K_array):
    
    print('K = ', K)
    
    resi_data, data = tools.get_resi_locs(files, K)
    
    ## uncomment this lines if you want to get all localizations to plot 
    # all_locs_x = np.append(np.array(resi_data['0']['x']),
    #                         np.array(resi_data['1']['x'])) * pxsize

    # all_locs_y = np.append(np.array(resi_data['0']['y']),
    #                        np.array(resi_data['1']['y'])) * pxsize
    
    # nclusters = resi_data['0']['cluster_id'].max()+1
    
    σ_resi_list = []
    
    if K == 1:
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
                
                # calculate covariance matrix
                cov = np.cov(cluster['x'] * pxsize, cluster['y'] * pxsize)

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

ax.legend()

# np.save('K_array', K_array)
# np.save('σ_resi_mean_array', σ_resi_mean_array)
# np.save('σ_resi_std_array', σ_resi_std_array)
    

    

    
    
