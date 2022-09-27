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

plt.close('all')

colors = ['#2880C4', '#97D8C4', '#F4B942', '#363636']
fig_knn, ax_knn = plt.subplots(figsize=(5, 5))

# =============================================================================
# Plot experimental data for comparison
# =============================================================================

#TODO: match experimental and simulated coordinates

path = r'well6_resting_RESI/'
filename = r'All_RESI_centers_noZ.hdf5'

filepath = os.path.join(path, filename)
df = pd.read_hdf(filepath, key = 'locs')

x = df.x*130
y = df.y*130

pos_exp = np.array([x, y]).T

from sklearn.neighbors import NearestNeighbors

### NN calculation ###
    
nbrs = NearestNeighbors(n_neighbors=5).fit(pos_exp) # find nearest neighbours
_distances_exp, _indices_exp = nbrs.kneighbors(pos_exp) # get distances and indices
# distances = _distances[:, 1] # get the first neighbour distances

colors = ['#2880C4', '#97D8C4', '#F4B942', '#363636']
# fig_knn, ax_knn = plt.subplots(figsize=(5, 5))

for i in range(4):

    # plot histogram of nn-distance of the simulation
    
    distances_exp = _distances_exp[:, i+1] # get the first neighbour distances
    
    bins = np.arange(0, 1000, 4)
    ax_knn.hist(distances_exp, bins=bins, alpha=0.5, color=colors[i], edgecolor='black', linewidth=0.1, density=True)

    ax_knn.set_xlim([0, 200])
    ax_knn.set_ylim([0, 0.025])
    
    ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
    ax_knn.set_ylabel('Frequency')
    ax_knn.tick_params(direction='in')
    ax_knn.set_box_aspect(1)
    
    plt.tight_layout()

ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
ax_knn.set_ylabel('Frequency')
ax_knn.tick_params(direction='in')
ax_knn.set_box_aspect(1)
    
# mask_area = mask.sum() * dx**2 

# print('Observed density = ', 1e6 * pos.shape[0]/mask_area, 'molecules per μm^2')

# =============================================================================
# plot 1st NN and compared to only monomers distribution
# =============================================================================
    
# colors = ['#2D7DD2']
# fig_1stnn, ax_1stnn = plt.subplots(figsize=(5, 5))

# for i in range(1):

#     # plot histogram of nn-distance of the simulation
    
#     distances = _distances[:, i+1] # get the first neighbour distances
    
#     freq, bins = np.histogram(distances, bins=200, density=True)
#     bin_centers = (bins[:-1] + bins[1:]) / 2
    
#     bins = np.arange(0, 1000, 4)
#     ax_1stnn.hist(_distances_exp[:, i+1], bins=bins, alpha=0.5, 
#                   color='#2880C4', edgecolor='black', linewidth=0.1, 
#                   density=True)
    
#     ax_1stnn.plot(bin_centers, freq, color=colors[i], linewidth=2, 
#                 label='uniform '+str(i+1)+'st-NN')
    
#     plt.tight_layout()
    
    
# ax_1stnn.set_xlim([0, 100])
# ax_1stnn.set_ylim([0, 0.022])

# ax_1stnn.set_xlabel('K-th nearest-neighbour distance (nm)')
# ax_1stnn.set_ylabel('Frequency')
# ax_1stnn.tick_params(direction='in')
# ax_1stnn.set_box_aspect(1)