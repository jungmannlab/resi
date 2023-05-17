#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:44:28 2022

@author: Luciano A. Masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors

plt.close('all')

# =============================================================================
# experimental parameters
# =============================================================================

# independent parameters

d = 2 # dimension of the simulation, d = 2 for 2D case, d = 3 for 3D
density = 100e-6 # molecules per nm^2 (or nm^3)
σ_dnapaint = 5 # nm
width = 40e3 # width of the simulated area in nm
height = 40e3 # height of the simulated area in nm
depth = 5e3 # depth of the simulated area in nm

# distribution = 'evenly spaced'
distribution = 'uniform'

# dependent parameters

resolution = 4 * σ_dnapaint # minimal distance between clusters to consider them resolvable
N = int(density * width * height)

# =============================================================================
# simulate molecules positions and calculate distances
# =============================================================================

pos = np.zeros((N, d)) # initialize array of localizations

if d == 2:
    
    fig0, ax0 = plt.subplots()
    
    if distribution == 'uniform':
        pos[:, 0], pos[:, 1] = [np.random.uniform(0, width, N), 
                                np.random.uniform(0, height, N)]
        
    elif distribution == 'evenly spaced':
        pos = np.mgrid[0:width:width/np.sqrt(N), 
                       0:height:height/np.sqrt(N)].reshape(2,-1).T
    
    ax0.scatter(pos[:, 0], pos[:, 1], alpha = 0.5)
    ax0.set_xlabel('x (nm)')
    ax0.set_ylabel('y (nm)')
    ax0.set_title('Density = '+str(int(density*1e6))+'/$μm^2$')
    
    ax0.set_xlim(0, 1000)
    ax0.set_ylim(0, 1000)
    ax0.tick_params(direction='in')
    ax0.set_box_aspect(1)

elif d == 3:
    
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(projection='3d')

    pos[:, 0], pos[:, 1], pos[:, 2] = [np.random.uniform(0, width, N), 
                                       np.random.uniform(0, height, N),
                                       np.random.uniform(0, depth, N)]
    
    ax0.scatter3D(pos[:, 0], pos[:, 1], pos[:, 2], edgecolor='none')
    ax0.set_xlabel('x (nm)')
    ax0.set_ylabel('y (nm)')
    ax0.set_zlabel('z (nm)')

nbrs = NearestNeighbors(n_neighbors=2).fit(pos) # find nearest neighbours
_distances, _indices = nbrs.kneighbors(pos) # get distances and indices
distances = _distances[:, 1] # get the first neighbour distances

# plot histogram of nn-distance
fig1, ax1 = plt.subplots()

bins = np.arange(0, 200, 5)

counts, *_ = ax1.hist(distances, bins=bins, alpha=0.5, edgecolor='black', linewidth=0.1, density=True)
ax1.set_xlabel('d_min (nm)')
ax1.set_ylabel('Frequency')
ax1.tick_params(direction='in')
ax1.set_box_aspect(1)

plt.tight_layout()

ax1.vlines(resolution, ymin=0, ymax=np.max(counts)+1, color='r', zorder=1)

ax1.set_xlim(0, 200)
ax1.set_ylim(0, 0.016)

n_subres = len(distances[distances < resolution])
n_subres_frac = n_subres/N

print(np.around(100 * n_subres_frac, 2), '% of molecules below the resolution limit of ', resolution, ' nm')
