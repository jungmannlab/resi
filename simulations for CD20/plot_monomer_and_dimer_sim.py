#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:40:32 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

filename = 'dataset1/well2_LTX_RESI_GFPNb_R1_400pM_2_MMStack_Pos0.ome_locs1000_RCC500_pick13_filter12_apicked.hdf5_varsD8_15.npz_RESI.npz'

data = dict(np.load(filename))

x = data['new_com_x_cluster']*130
y = data['new_com_y_cluster']*130

pos_exp = np.array([x, y]).T

from sklearn.neighbors import NearestNeighbors

### NN calculation ###
    
nbrs = NearestNeighbors(n_neighbors=5).fit(pos_exp) # find nearest neighbours
_distances_exp, _indices_exp = nbrs.kneighbors(pos_exp) # get distances and indices
# distances = _distances[:, 1] # get the first neighbour distances

# =============================================================================
# plot 1st NN and compared to only monomers distribution
# =============================================================================
    
color = '#4059AD'
fig_1stnn, ax_1stnn = plt.subplots(figsize=(5, 5))

data_dim_and_mon = np.load('data_dim_and_mon.npy')
data_mon = np.load('data_only_mon.npy')

freq_dim_and_mon, bin_centers_dim_and_mon = data_dim_and_mon[1, :], data_dim_and_mon[0, :]

ax_1stnn.plot(bin_centers_dim_and_mon, freq_dim_and_mon, color=color, 
              linewidth=2,label='uniform '+str(1)+'st-NN')

freq_mon, bin_centers_mon = data_mon[1, :], data_mon[0, :]

ax_1stnn.plot(bin_centers_mon, freq_mon, color=color, 
              linewidth=2,label='uniform '+str(1)+'st-NN', linestyle='dashed', alpha=.6)
    
bins = np.arange(0, 1000, 2)
ax_1stnn.hist(_distances_exp[:, 1], bins=bins, alpha=0.7, color=color, 
             edgecolor='black', linewidth=0.1, density=True)
    
plt.tight_layout()
    
    
ax_1stnn.set_xlim([0, 100])
ax_1stnn.set_ylim([0, 0.022])

ax_1stnn.set_xlabel('K-th nearest-neighbour distance (nm)')
ax_1stnn.set_ylabel('Frequency')
ax_1stnn.tick_params(direction='in')
ax_1stnn.set_box_aspect(1)