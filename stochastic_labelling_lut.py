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
density_arr = np.linspace(1, 100, 100)* 10**-6 # molecules per nm^2
σ_dnapaint_arr = np.linspace(1, 20, 100) # nm
labelling_rounds = np.arange(1, 10)
width = 40e3 # width of the simulated area in nm
height = 40e3 # height of the simulated area in nm

# dependent parameters

resolution_arr = 4 * σ_dnapaint_arr # minimal distance between clusters to consider them resolvable
N = np.array(density_arr * width * height, dtype=int)

# =============================================================================
# simulate localizations
# =============================================================================

n_subres_frac_arr = np.zeros((len(density_arr), len(resolution_arr)))

for i, density in enumerate(density_arr):
    
    # calculate number of molecules
    N = int(density * width * height)

    # simulate positons for the molecules
    pos = np.zeros((N, d)) # initialize array of localizations
    pos[:, 0], pos[:, 1] = [np.random.uniform(0, width, N), 
                             np.random.uniform(0, height, N)]
    
    nbrs = NearestNeighbors(n_neighbors=2).fit(pos) # find nearest neighbours
    _distances, _indices = nbrs.kneighbors(pos) # get distances and indices
    distances = _distances[:, 1] # get only the first neighbour distances
    
    for j, resolution in enumerate(resolution_arr):
    
        n_subres = len(distances[distances < resolution])
        n_subres_frac = n_subres/N
        
        n_subres_frac_arr[i, j] = n_subres_frac
        

fig, ax = plt.subplots()
plot = ax.imshow(n_subres_frac_arr, interpolation='None', origin='lower')
fig.colorbar(plot, ax=ax)

ax.set_ylim(density_arr[0], density_arr[-1])
ax.set_xlim(resolution_arr[0], resolution_arr[-1])

