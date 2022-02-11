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
density_arr = np.linspace(1, 1000, 300) * 10**-6 # molecules per nm^2
σ_dnapaint_arr = np.linspace(1, 20, 100) # nm
# labelling_rounds = np.arange(1, 10)
width = 80e3 # width of the simulated area in nm
height = 80e3 # height of the simulated area in nm
distribution = 'uniform'

err_val = 0.05 # admitted frac of molecules closer than the resolution limit

# dependent parameters

resolution_arr = 4 * σ_dnapaint_arr # minimal distance between clusters to consider them resolvable
N = np.array(density_arr * width * height, dtype=int)

# =============================================================================
# simulate molecule positions
# =============================================================================

n_subres_frac_arr = np.zeros((len(density_arr), len(resolution_arr)))

for i, density in enumerate(density_arr):
    
    # calculate number of molecules
    N = int(density * width * height)

    # simulate positons for the molecules
    pos = np.zeros((N, d)) # initialize array of localizations
    
    if distribution == 'uniform':
    
        pos = np.array([np.random.uniform(0, width, N), 
                        np.random.uniform(0, height, N)]).T
    
    elif distribution == 'evenly spaced':
        
        wstep = width/np.sqrt(N)
        hstep = height/np.sqrt(N)
        pos = np.mgrid[0:width:wstep, 
                       0:height:hstep].reshape(2,-1).T
    
    
    nbrs = NearestNeighbors(n_neighbors=2).fit(pos) # find nearest neighbours
    _distances, _indices = nbrs.kneighbors(pos) # get distances and indices
    distances = _distances[:, 1] # get only the first neighbour distances
    
    for j, resolution in enumerate(resolution_arr):
    
        n_subres = len(distances[distances < resolution])
        n_subres_frac = n_subres/N
        
        n_subres_frac_arr[i, j] = n_subres_frac
        

err_val_arr = n_subres_frac_arr // err_val + 1 
# TODO: solve in general, now only valid for linear regime of uniform distr

# =============================================================================
# 2D plot frac of molecules closer than res limit vs (resolution, density)
# =============================================================================

fig0, ax0 = plt.subplots()
ax0.set_title(str(distribution)+' distribution, err_val = ' + str(err_val))
extent = [resolution_arr[0], resolution_arr[-1], 
          density_arr[0] * 10**6, density_arr[-1] * 10**6] # convert to μm^2
plot = ax0.imshow(n_subres_frac_arr, interpolation='None', extent=extent,
                 origin='lower', aspect='auto')
cbar0 = fig0.colorbar(plot, ax=ax0)
cbar0.set_label('Fraction of molecules closer than resolution limit')

ax0.set_xlabel('Resolution ($nm$)')
ax0.set_ylabel('Density ($μm^{-2}$)')

# fig1, ax1 = plt.subplots()
# ax1.set_title(str(distribution)+' distribution, err_val = ' + str(err_val))
# extent = [resolution_arr[0], resolution_arr[-1], 
#           density_arr[0] * 10**6, density_arr[-1] * 10**6] # convert to μm^2
# plot = ax1.imshow(err_val_arr, interpolation='None', extent=extent,
#                  origin='lower', aspect='auto')
# cbar1 = fig1.colorbar(plot, ax=ax1)
# cbar1.set_label('Number of imaging rounds needed')

# ax1.set_xlabel('Resolution ($nm$)')
# ax1.set_ylabel('Density ($μm^{-2}$)')

# =============================================================================
# 1D plot frac of molecules closer than res limit vs density (resolution fixed)
# =============================================================================

fig2, ax2 = plt.subplots()

res_id = np.array(np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * len(resolution_arr), dtype=int)
ax2.plot(density_arr * 10**6, n_subres_frac_arr[:, res_id[3]])
# title = 'Resolution = ' + str(np.around(resolution_arr[res_id], 1)) + ' nm'
# ax2.set_title(title)
ax2.set_xlabel('Density ($μm^{-2}$)')
ax2.set_ylabel('Fraction of subres molecules')
ax2.plot(density_arr * 10**6, np.ones(len(density_arr)) * err_val, 'k--')

# res_id = int(0.1 * len(resolution_arr)) # will give roughly 20 nm res
ax2.plot(density_arr * 10**6, n_subres_frac_arr[:, res_id[3]])

for index in res_id:
    
    ax2.plot(density_arr * 10**6, n_subres_frac_arr[:, index], 
             label=str(np.around(resolution_arr[index], 0))+' nm')
    
ax2.legend()