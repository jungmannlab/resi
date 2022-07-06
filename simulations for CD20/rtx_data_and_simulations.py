#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:02:36 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multinomial, truncnorm
from scipy.special import comb

# =============================================================================
# plot experimental data of RTX CD20
# =============================================================================

fig_knn, ax_knn = plt.subplots(figsize=(12,6))

ax_knn.set_xlim([0, 100])
ax_knn.set_ylim([0, 0.055])
ax_knn.set_xlabel('Kth NN distance (nm)')
ax_knn.set_ylabel('Norm Frequency')

colors = ['#4059AD', '#97D8C4', '#F4B942', '#363636']

ax_knn.tick_params(direction='in')

plt.tight_layout()


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

colors = ['#2880C4', '#97D8C4', '#F4B942', '#363636']
# fig_knn, ax_knn = plt.subplots(figsize=(5, 5))

for i in range(4):

    # plot histogram of nn-distance of the simulation
    
    distances = _distances[:, i+1] # get the first neighbour distances
    
    # freq, bins = np.histogram(distances, bins=100, density=True)
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # ax_knn.plot(bin_centers, freq, color=colors[i], linewidth=2, 
    #             label='uniform '+str(i+1)+'st-NN')
    
    bins = np.arange(0, 1000, 2)
    ax_knn.hist(distances, bins=bins, alpha=0.3, color=colors[i], edgecolor='black', linewidth=0.1, density=True)

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


# =============================================================================
# simulation of complete spatial random distribution of CD20
# =============================================================================

from sklearn.neighbors import NearestNeighbors


# experimental parameters

# independent parameters

d = 2 # dimension of the simulation, d = 2 for 2D case, d = 3 for 3D
density = 106e-6 # molecules per nm^2 (or nm^3)
σ_dnapaint = 5 # nm
width = 300e3 # width of the simulated area in nm
height = 300e3 # height of the simulated area in nm
depth = 5e3 # depth of the simulated area in nm

# distribution = 'evenly spaced'
distribution = 'uniform'

# dependent parameters

resolution = 4 * σ_dnapaint # minimal distance between clusters to consider them resolvable
N = int(density * width * height)

pos = np.zeros((N, d)) # initialize array of localizations

if d == 2:
    
    # fig0, ax0 = plt.subplots()
    
    if distribution == 'uniform':
        pos[:, 0], pos[:, 1] = [np.random.uniform(0, width, N), 
                                np.random.uniform(0, height, N)]
        
    elif distribution == 'evenly spaced':
        pos = np.mgrid[0:width:width/np.sqrt(N), 
                       0:height:height/np.sqrt(N)].reshape(2,-1).T
    
    # ax0.scatter(pos[:, 0], pos[:, 1], alpha = 0.5)
    # ax0.set_xlabel('x (nm)')
    # ax0.set_ylabel('y (nm)')
    # ax0.set_title('Density = '+str(int(density*1e6))+'/$μm^2$')
    
nbrs = NearestNeighbors(n_neighbors=5).fit(pos) # find nearest neighbours
_distances, _indices = nbrs.kneighbors(pos) # get distances and indices
distances = _distances[:, 1] # get the first neighbour distances

print('max distance', np.max(distances))
print('min distance', np.min(distances))

distances_save = distances.copy()

# plot histogram of nn-distance
# fig1, ax1 = plt.subplots()
freq_uniform, bins_uniform = np.histogram(distances, bins=2000, density=True)
bin_centers_uniform = (bins_uniform[:-1] + bins_uniform[1:]) / 2

# ax.plot(bin_centers_uniform, freq_uniform, color='g', linewidth=0.5, 
#         linestyle='dotted', label='uniform 1st-NN')

# ax1.set_xlabel('d_min (nm)')
# ax1.set_ylabel('Counts')
# ax1.set_xlim([0, 100])

# ax.legend()


for i in range(4):

    # plot histogram of nn-distance
    
    distances = _distances[:, i+1] # get the first neighbour distances
    
    freq_uniform, bins_uniform = np.histogram(distances, bins=1000, density=True)
    
    bin_centers_uniform = (bins_uniform[:-1] + bins_uniform[1:]) / 2
    
    ax_knn.plot(bin_centers_uniform, freq_uniform, color=colors[i], linewidth=2, 
                label='uniform '+str(i+1)+'st-NN')
    

    
    plt.tight_layout()