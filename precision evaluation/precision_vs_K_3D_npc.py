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

# files = [r'/Volumes/pool-miblab4/users/masullo/z.microscopy_processed/resi/sample_pick/R1_7nt_150pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0_apicked.hdf5', 
#          r'/Volumes/pool-miblab4/users/masullo/z.microscopy_processed/resi/sample_pick/R2_7nt_100pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0_apicked.hdf5',
#          r'/Volumes/pool-miblab4/users/masullo/z.microscopy_processed/resi/sample_pick/R3_7nt_100pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0_apicked.hdf5',
#          r'/Volumes/pool-miblab4/users/masullo/z.microscopy_processed/resi/sample_pick/R4_7nt_100pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0_apicked.hdf5']

files = [r'W:/users/masullo/z.microscopy_processed/resi/aligned/R1_7nt_150pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0.hdf5', 
         r'W:/users/masullo/z.microscopy_processed/resi/aligned/R2_7nt_100pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0.hdf5',
         r'W:/users/masullo/z.microscopy_processed/resi/aligned/R3_7nt_100pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0.hdf5',
         r'W:/users/masullo/z.microscopy_processed/resi/aligned/R4_7nt_100pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0.hdf5']

pxsize = 130 # nm
σ_dnapaint = 4.5
σ_z_dnapaint = 2 * σ_dnapaint

# K_array = np.array([1, 2, 3, 5, 10, 20, 30, 40, 60, 80, 100]) # number of localizations per subset
K_array = np.array([2, 10, 20]) # number of localizations per subset
minsize = 5 # minimum number of resi localizations needed to consider the statistics
σ_resi_mean_array = np.zeros(len(K_array))
σ_resi_std_array = np.zeros(len(K_array))
σ_z_resi_mean_array = np.zeros(len(K_array))
σ_z_resi_std_array = np.zeros(len(K_array))
counter = np.zeros(len(K_array)) # counter to keep track of how many σ will be considered for each K


# iterate over the array of different K values
for k, K in enumerate(K_array):
    
    print('K = ', K)
    
    resi_data, data = tools.get_resi_locs(files, K)
            
    dataframes = [resi_data['0'], resi_data['1'], resi_data['2'], resi_data['3']]
    merged_resi = pd.concat(dataframes)
    
    path = r'W:/users/masullo/z.microscopy_processed/resi'
    oldname = r'R1_7nt_150pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0.hdf5'
    path = r'W:/users/masullo/z.microscopy_processed/resi/aligned/'
    tools.picasso_hdf5(merged_resi, 'npc_resampling_K' + str(K) + '_.hdf5', 
                       oldname, path)
    
    ## uncomment this lines if you want to get all localizations to plot 
    # all_locs_x = np.concatenate((np.array(resi_data['0']['x']),
    #                             np.array(resi_data['1']['x']),
    #                             np.array(resi_data['2']['x']),
    #                             np.array(resi_data['3']['x']))) * pxsize

    # all_locs_y = np.concatenate((np.array(resi_data['0']['y']),
    #                             np.array(resi_data['1']['y']),
    #                             np.array(resi_data['2']['y']),
    #                             np.array(resi_data['3']['y']))) * pxsize
    
    # all_locs_z = np.concatenate((np.array(resi_data['0']['z']),
    #                             np.array(resi_data['1']['z']),
    #                             np.array(resi_data['2']['z']),
    #                             np.array(resi_data['3']['z']))) * pxsize
    
    # np.save('all_locs_x_K' + str(K) , all_locs_x)
    # np.save('all_locs_y_K' + str(K) , all_locs_y)
    # np.save('all_locs_z_K' + str(K) , all_locs_z)

    σ_resi_list = []
    σ_z_resi_list = []
    
    # get the clusters id without repetitions (they can be any set of labels)
    cluster_id_set = list(set(list(resi_data['0']['cluster_id'])))
    
    for key in ['0']:
        for i, cluster_id in enumerate(cluster_id_set):
            
            # print(cluster_id)
        
            # get each cluster localizations
            cluster = resi_data[key].loc[resi_data[key]['cluster_id'] == cluster_id]
            cluster_size = cluster.shape[0]
            
            if cluster_size > minsize:
                
                # calculate covariance matrix
                cov = np.cov(cluster['x'] * pxsize, cluster['y'] * pxsize)

                # calculate 1D σ
                σ_resi = np.sqrt(1/2 * np.trace(cov))
                σ_z_resi = np.std(cluster['z']) 
                
                # save σ_resi into the list
                σ_resi_list.append(σ_resi)
                σ_z_resi_list.append(σ_z_resi)
                
                counter[k] += 1
    
    # calculate mean, std and save σ_mean and σ_std for every K
    σ_resi_mean = np.array(σ_resi_list).mean() 
    σ_resi_std = np.array(σ_resi_list).std()
     
    σ_resi_mean_array[k] = σ_resi_mean
    σ_resi_std_array[k] = σ_resi_std
    
    # calculate mean, std and save σ_z_mean and σ_z_std for every K
    σ_z_resi_mean = np.array(σ_z_resi_list).mean() 
    σ_z_resi_std = np.array(σ_z_resi_list).std()
     
    σ_z_resi_mean_array[k] = σ_z_resi_mean
    σ_z_resi_std_array[k] = σ_z_resi_std

print('Number of valid clusters (size > 5 locs) for each K are: ', counter)

# plot results
     
fig, ax = plt.subplots(figsize =(5, 5))

ax.errorbar(K_array, σ_resi_mean_array, yerr=σ_resi_std_array, fmt='ko', 
            capsize=5, label='Measured data', zorder=2)

ax.plot(np.arange(1, 35, 0.2), σ_dnapaint/np.arange(1, 35, 0.2)**(1/2), 
        color='#266DD3', label='$σ_{dna-paint} / K^{1/2}$')

ax.set_xlabel('K')
ax.set_ylabel('$σ_{resi}$ (nm)')

ax.legend()

fig1, ax1 = plt.subplots(figsize =(5, 5))

ax1.errorbar(K_array, σ_z_resi_mean_array, yerr=σ_z_resi_std_array, fmt='ko', 
            capsize=5, label='Measured data', zorder=2)

ax1.plot(np.arange(1, 35, 0.2), σ_z_dnapaint/np.arange(1, 35, 0.2)**(1/2), 
        color='#266DD3', label='$σ_{z-dna-paint} / K^{1/2}$')

ax1.set_xlabel('K')
ax1.set_ylabel('$σ_{z-resi}$ (nm)')

ax1.legend()
    

    

    
    