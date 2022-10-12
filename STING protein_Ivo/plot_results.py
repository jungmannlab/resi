#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:54:01 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import configparser
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

# read the metadata
config = configparser.ConfigParser()
config.read('K2_picked_mask_in_resi_7_15_MASK_nn_distances_params.txt')
params = config['params']

# read the simulated nnd
sim_nnd = np.load('K2_picked_mask_in_resi_7_15_MASK_nn_distances.npy')

# sim_nnd = np.load('linear_masks_nn_distances.npy')

sim_nnd = np.load('circular_masks_nn_distances.npy')


K = int(params['k nn'])

# =============================================================================
# Plot simulated data
# =============================================================================

colors = ['#2880C4', '#97D8C4', '#F4B942', '#363636']
fig_knn, ax_knn = plt.subplots(figsize=(5, 5))
fig_knn.suptitle('NN distances')

for i in range(K):
    
    # plot histogram of nn-distance of the simulation
        
    distances = sim_nnd[:, i+1] # get the first neighbour distances
        
    freq, bins = np.histogram(distances, bins=200, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax_knn.plot(bin_centers, freq, color=colors[i], linewidth=2, 
                label='uniform '+str(i+1)+'st-NN')
        
    # ax_knn.set_xlim([0, 100])
    # ax_knn.set_ylim([0, 0.022])
        
    ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
    ax_knn.set_ylabel('Frequency')
    ax_knn.tick_params(direction='in')
    ax_knn.set_box_aspect(1)
    
# =============================================================================
# Plot experimental data for comparison
# =============================================================================

# Ivo STING protein

# path = r'well6_resting_RESI/'
# filename = r'All_RESI_centers_noZ.hdf5'

# Susi FA

path = r'forLuciano/'
filename = r'K2_picked_mask_in_resi_7_15.hdf5'

filepath = os.path.join(path, filename)
df = pd.read_hdf(filepath, key = 'locs')

cam_px_size = 130 # in nm

x = df.x * cam_px_size 
y = df.y * cam_px_size 

pos_exp = np.array([x, y]).T

### NN calculation ###
    
nbrs = NearestNeighbors(n_neighbors=5).fit(pos_exp) # find nearest neighbours
_distances_exp, _indices_exp = nbrs.kneighbors(pos_exp) # get distances and indices
# distances = _distances[:, 1] # get the first neighbour distances

colors = ['#2880C4', '#97D8C4', '#F4B942', '#363636']
# fig_knn, ax_knn = plt.subplots(figsize=(5, 5))

for i in range(4):

    # plot histogram of nn-distance of the simulation
    
    distances_exp = _distances_exp[:, i+1] # get the first neighbour distances
    
    bins = np.arange(0, 1000, 1)
            
    ax_knn.hist(distances_exp, bins=bins, alpha=0.5, color=colors[i], edgecolor='black', linewidth=0.1, density=True)

    ax_knn.set_xlim([0, 200])
    ax_knn.set_ylim([0, 0.05])
    
    ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
    ax_knn.set_ylabel('Frequency')
    ax_knn.tick_params(direction='in')
    ax_knn.set_box_aspect(1)
    
    plt.tight_layout()
    
# =============================================================================
# plot 1st NN and compared to only monomers distribution
# =============================================================================
    
colors = ['#2D7DD2']
fig_1stnn, ax_1stnn = plt.subplots(figsize=(5, 5))
fig_1stnn.suptitle('1st NN distances')

for i in range(1):

    # plot histogram of nn-distance of the simulation

    distances = sim_nnd[:, i+1] # get the first neighbour distances
    
    freq, bins = np.histogram(distances, bins=200, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bins = np.arange(0, 1000, 4)
    ax_1stnn.hist(_distances_exp[:, i+1], bins=bins, alpha=0.5, 
                  color='#2880C4', edgecolor='black', linewidth=0.1, 
                  density=True)
    
    ax_1stnn.plot(bin_centers, freq, color=colors[i], linewidth=2, 
                label='uniform '+str(i+1)+'st-NN')
    
    plt.tight_layout()
    
    
ax_1stnn.set_xlim([0, 200])
ax_1stnn.set_ylim([0, 0.05])

ax_1stnn.set_xlabel('K-th nearest-neighbour distance (nm)')
ax_1stnn.set_ylabel('Frequency')
ax_1stnn.tick_params(direction='in')
ax_1stnn.set_box_aspect(1)