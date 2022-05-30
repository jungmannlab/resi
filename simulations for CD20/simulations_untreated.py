#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:48:42 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# =============================================================================
# experimental parameters
# =============================================================================

# independent parameters

d = 2 # dimension of the simulation, d = 2 for 2D case, d = 3 for 3D
mult = 2 # multiplicity of the molecular assembly (e.g. mult = 2 for dimers)
D_dimer = 6
density = 50e-6 # molecules per nm^2 (or nm^3)
σ_label = 5 # nm
width = 80e3 # width of the simulated area in nm
height = 80e3 # height of the simulated area in nm
depth = 5e3 # depth of the simulated area in nm

# distribution = 'evenly spaced'
distribution = 'uniform'

# dependent parameters

# resolution = 4 * σ_dnapaint # minimal distance between clusters to consider them resolvable
N = int(density * width * height)

# =============================================================================
# simulate molecules positions and calculate distances
# =============================================================================

c_pos = np.zeros((N, d)) # initialize array of central positions

if d == 2:
    
    fig0, ax0 = plt.subplots()
    
    if distribution == 'uniform':
        c_pos[:, 0], c_pos[:, 1] = [np.random.uniform(0, width, N), 
                                    np.random.uniform(0, height, N)]
        
    elif distribution == 'evenly spaced':
        c_pos = np.mgrid[0:width:width/np.sqrt(N), 
                         0:height:height/np.sqrt(N)].reshape(2,-1).T
        
    ax0.scatter(c_pos[:, 0], c_pos[:, 1], alpha=0.5)
    
    ax0.set_xlabel('x (nm)')
    ax0.set_ylabel('y (nm)')
    ax0.set_title('Density = '+str(int(density*1e6))+'/$μm^2$')
    ax0.set_box_aspect(1)

angle = np.random.uniform(0, 2*np.pi, N)
D0 = np.random.normal(loc=D_dimer, scale=σ_label, size=N)
D1 = np.random.normal(loc=D_dimer, scale=σ_label, size=N)

pos = np.zeros((N, d, mult))

pos[:, :, 0] = c_pos + np.array([D0*np.cos(angle), D0*np.sin(angle)]).T
pos[:, :, 1] = c_pos - np.array([D1*np.cos(angle), D1*np.sin(angle)]).T

# this plot should output dimers with its center, and two molecules marked with different colors
ax0.scatter(pos[:, :, 0][:, 0], pos[:, :, 0][:, 1], alpha=0.5)
ax0.scatter(pos[:, :, 1][:, 0], pos[:, :, 1][:, 1], alpha=0.5)

        
# =============================================================================
# nn calculation
# =============================================================================

from sklearn.neighbors import NearestNeighbors

# flatten the array to get all molecules positions together
pos = np.concatenate((pos[:, :, 0], pos[:, :, 1]), axis=0) 

no_dimer_trick = False
if no_dimer_trick:
    pos = c_pos
    mult = 1

np.random.shuffle(pos)
print(pos.shape)
# this plot should output dimers without its center, and two molecules marked with the same color
fig1, ax1 = plt.subplots()

ax1.scatter(pos[:, 0], pos[:, 1], alpha=0.5)

ax1.set_xlabel('x (nm)')
ax1.set_ylabel('y (nm)')
ax1.set_title('Density = '+str(int(density*1e6))+'/$μm^2$')
ax1.set_box_aspect(1)

# labelling correction
labelling = True
p = 0.5

if labelling:
    
    ids = np.random.choice(np.arange(mult*N), size=int((mult*N)*p), replace=False)
    
    pos = pos[ids]
    
    
# this plot should output dimers taking into account labelling

ax1.scatter(pos[:, 0], pos[:, 1], facecolors='none', edgecolors='k')

# nn calculation
    
nbrs = NearestNeighbors(n_neighbors=5).fit(pos) # find nearest neighbours
_distances, _indices = nbrs.kneighbors(pos) # get distances and indices
# distances = _distances[:, 1] # get the first neighbour distances

colors = ['#4059AD', '#97D8C4', '#F4B942', '#363636']
fig_knn, ax_knn = plt.subplots(figsize=(5, 5))

for i in range(4):

    # plot histogram of nn-distance
    
    distances = _distances[:, i+1] # get the first neighbour distances
    
    freq, bins = np.histogram(distances, bins=10000, density=True)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax_knn.plot(bin_centers, freq, color=colors[i], linewidth=2, 
                label='uniform '+str(i+1)+'st-NN')
    
    plt.tight_layout()
    
ax_knn.set_xlim([0, 100])

ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
ax_knn.set_ylabel('Frequency')
ax_knn.tick_params(direction='in')
ax_knn.set_box_aspect(1)
    