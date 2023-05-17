#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:48:08 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

# filename = 'dataset2/well2_PIB2_neg_GFPNb_R1_200pM_1_MMStack_Pos0.ome_locs500_RCC500_pick20_filter12_apicked_varsD8_20.npz_RESI.npz'

# filename = 'dataset1/well2_LTX_RESI_GFPNb_R1_400pM_2_MMStack_Pos0.ome_locs1000_RCC500_pick13_filter12_apicked.hdf5_varsD8_15.npz_RESI.npz'

filename = 'NND data RTX/well2_CHO_A3_GFPAlfaCd20_RTXAlexa647_GFPNb_apicked_varsD9_15.npz_RESI.npz'

data = dict(np.load(filename))

x = data['new_com_x_cluster']*130
y = data['new_com_y_cluster']*130

pos = np.array([x, y]).T

from sklearn.neighbors import NearestNeighbors

### NN calculation ###
    
nbrs = NearestNeighbors(n_neighbors=6).fit(pos) # find nearest neighbours
_distances, _indices = nbrs.kneighbors(pos) # get distances and indices
# distances = _distances[:, 1] # get the first neighbour distances

colors = ['#4059AD', '#97D8C4', '#F4B942', '#363636', 'r']
fig_knn, ax_knn = plt.subplots(figsize=(5, 5))

for i in range(5):

    # plot histogram of nn-distance of the simulation
    
    distances = _distances[:, i+1] # get the first neighbour distances
    
    # freq, bins = np.histogram(distances, bins=100, density=True)
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # ax_knn.plot(bin_centers, freq, color=colors[i], linewidth=2, 
    #             label='uniform '+str(i+1)+'st-NN')
    
    bins = np.arange(0, 1000, 2)
    ax_knn.hist(distances, bins=bins, alpha=0.7, color=colors[i], edgecolor='black', linewidth=0.1, density=True)

    # ax_knn.set_xlim([0, 200])
    # ax_knn.set_ylim([0, 0.022])
    
    ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
    ax_knn.set_ylabel('Frequency')
    ax_knn.tick_params(direction='in')
    ax_knn.set_box_aspect(1)
    
    plt.tight_layout()
    
# ax_knn.set_xlim([0, 100])
# ax_knn.set_ylim([0, 0.022])

ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
ax_knn.set_ylabel('Frequency')
ax_knn.tick_params(direction='in')
ax_knn.set_box_aspect(1)