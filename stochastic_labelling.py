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
density = 100e-6 # molecules per nm^2
σ_dnapaint = 5 # nm
labelling_rounds = 4
width = 20e3 # width of the simulated area in nm
height = 20e3 # height of the simulated area in nm

# dependent parameters

resolution = 4 * σ_dnapaint # minimal distance between clusters to consider them resolvable
N = int(density * width * height)

# =============================================================================
# simulate molecules positions and calculate distances
# =============================================================================

pos = np.zeros((N, d)) # initialize array of localizations
pos[:, 0], pos[:, 1] = [np.random.uniform(0, width, N), 
                         np.random.uniform(0, height, N)]

fig0, ax0 = plt.subplots()

ax0.scatter(pos[:, 0], pos[:, 1], alpha = 0.5)

nbrs = NearestNeighbors(n_neighbors=2).fit(pos) # find nearest neighbours
_distances, _indices = nbrs.kneighbors(pos) # get distances and indices
distances = _distances[:, 1] # get the first neighbour distances

# plot histogram of nn-distance
fig1, ax1 = plt.subplots()
counts, *_ = ax1.hist(distances, bins=50, alpha=0.5)
ax1.set_xlabel('d_min (nm)')
ax1.set_ylabel('Counts')

ax1.vlines(resolution, ymin=0, ymax=np.max(counts), color='r', zorder=1)

n_subres = len(distances[distances < resolution])
n_subres_frac = n_subres/N

print(np.around(100 * n_subres_frac, 2), '% of molecules below the resolution limit of ', resolution, ' nm')
