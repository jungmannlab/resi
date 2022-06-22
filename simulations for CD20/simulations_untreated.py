#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:48:42 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

# =============================================================================
# experimental parameters
# =============================================================================

# independent parameters

D = 2 # dimension of the simulation, d = 2 for 2D case, d = 3 for 3D
mult = 2 # multiplicity of the molecular assembly (e.g. mult = 2 for dimers)
D_dimer = 6
density_d = 50e-6 # molecules per nm^2 (or nm^3)
density_m = 50e-6 # molecules per nm^2 (or nm^3)

σ_label = 5 # nm
width = 80e3 # width of the simulated area in nm
height = 80e3 # height of the simulated area in nm
depth = 5e3 # depth of the simulated area in nm

dim_color = '#009FB7'
mon_color = '#FE4A49'

# distribution = 'evenly spaced'
distribution = 'uniform'

# labelling correction
labelling = True
p = 0.5

# dependent parameters

# resolution = 4 * σ_dnapaint # minimal distance between clusters to consider them resolvable
N_d = int(density_d/2 * width * height) # divided by two because it's the number of centers of dimers
N_m = int(density_m * width * height)

# =============================================================================
# simulate molecules positions and calculate distances
# =============================================================================

c_pos_dim = np.zeros((N_d, D)) # initialize array of central positions for dimers
c_pos_mon = np.zeros((N_m, D)) # initialize array of central positions for monomers

if D == 2:
    
    fig0, ax0 = plt.subplots() # dimers
    fig0.suptitle('Dimers + their center')
    fig1, ax1 = plt.subplots() # monomers
    fig1.suptitle('Monomers')
    
    if distribution == 'uniform':
        c_pos_dim[:, 0], c_pos_dim[:, 1] = [np.random.uniform(0, width, N_d), 
                                            np.random.uniform(0, height, N_d)]
        
        c_pos_mon[:, 0], c_pos_mon[:, 1] = [np.random.uniform(0, width, N_m), 
                                            np.random.uniform(0, height, N_m)]
        
    elif distribution == 'evenly spaced':
        c_pos_dim = np.mgrid[0:width:width/np.sqrt(N_d), 
                             0:height:height/np.sqrt(N_d)].reshape(2,-1).T
        
        #TODO: complete for monomers
        
    ax0.scatter(c_pos_dim[:, 0], c_pos_dim[:, 1], alpha=0.5, marker='*')
    
    ax0.set_xlabel('x (nm)')
    ax0.set_ylabel('y (nm)')
    ax0.set_title('Real density = '+str(int(density_d*1e6))+'/$μm^2$')
    ax0.set_box_aspect(1)
    
    ax1.scatter(c_pos_mon[:, 0], c_pos_mon[:, 1], alpha=0.5, color=mon_color)
    
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    ax1.set_title('Real density = '+str(int(density_m*1e6))+'/$μm^2$')
    ax1.set_box_aspect(1)

angle = np.random.uniform(0, 2*np.pi, N_d)
D0 = np.random.normal(loc=D_dimer, scale=σ_label, size=N_d)
D1 = np.random.normal(loc=D_dimer, scale=σ_label, size=N_d)

pos_dim = np.zeros((N_d, D, mult))

pos_dim[:, :, 0] = c_pos_dim + np.array([D0*np.cos(angle), D0*np.sin(angle)]).T
pos_dim[:, :, 1] = c_pos_dim - np.array([D1*np.cos(angle), D1*np.sin(angle)]).T

# this plot should output dimers with its center, and two molecules marked with different colors
ax0.scatter(pos_dim[:, :, 0][:, 0], pos_dim[:, :, 0][:, 1], alpha=0.5, 
            color=dim_color )
ax0.scatter(pos_dim[:, :, 1][:, 0], pos_dim[:, :, 1][:, 1], alpha=0.5, 
            color=dim_color )

length = 1000 # nm, length of the display area

ax0.set_xlim(width/2, width/2 + length)
ax0.set_ylim(width/2, width/2 + length)

ax1.set_xlim(width/2, width/2 + length)
ax1.set_ylim(width/2, width/2 + length)

        
# =============================================================================
# nn calculation
# =============================================================================

from sklearn.neighbors import NearestNeighbors

# flatten the array to get all molecules positions together
pos_dim = np.concatenate((pos_dim[:, :, 0], pos_dim[:, :, 1]), axis=0) 
pos_mon = c_pos_mon # for monomers the array stays the same

no_dimer_trick = False
if no_dimer_trick:
    pos_dim = c_pos_dim
    mult = 1

np.random.shuffle(pos_dim)

# this plot should output dimers without its center, 
# two molecules marked with the same color AND the monomers in another color
fig2, ax2 = plt.subplots()
fig2.suptitle('Monomers + dimers')

ax2.scatter(pos_dim[:, 0], pos_dim[:, 1], alpha=0.5, color=dim_color)
ax2.scatter(pos_mon[:, 0], pos_mon[:, 1], alpha=0.5, color=mon_color)

ax2.set_xlabel('x (nm)')
ax2.set_ylabel('y (nm)')
ax2.set_title('Real density = '+str(int((density_d + density_m)*1e6))+'/$μm^2$')
ax2.set_box_aspect(1)

ax2.set_xlim(width/2, width/2 + length)
ax2.set_ylim(width/2, width/2 + length)


pos = np.concatenate((pos_dim, pos_mon)) 

np.random.shuffle(pos)

N = mult*N_d + N_m

if labelling:
    
    ids = np.random.choice(np.arange(N), size=int((N)*p), 
                           replace=False)
    
    pos = pos[ids]
    
print(pos.shape)

# this plot should output dimers taking into account labelling
ax2.scatter(pos[:, 0], pos[:, 1], facecolors='none', edgecolors='k')

# molecules with black edge are the ones actually labelled

# nn calculation
    
nbrs = NearestNeighbors(n_neighbors=5).fit(pos) # find nearest neighbours
_distances, _indices = nbrs.kneighbors(pos) # get distances and indices
# distances = _distances[:, 1] # get the first neighbour distances

colors = ['#4059AD', '#97D8C4', '#F4B942', '#363636']
fig_knn, ax_knn = plt.subplots(figsize=(5, 5))

for i in range(4):

    # plot histogram of nn-distance
    
    distances = _distances[:, i+1] # get the first neighbour distances
    
    freq, bins = np.histogram(distances, bins=1000, density=True)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax_knn.plot(bin_centers, freq, color=colors[i], linewidth=2, 
                label='uniform '+str(i+1)+'st-NN')
    
    plt.tight_layout()
    
ax_knn.set_xlim([0, 100])
ax_knn.set_ylim([0, 0.022])

ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
ax_knn.set_ylabel('Frequency')
ax_knn.tick_params(direction='in')
ax_knn.set_box_aspect(1)

### plot experimental data

colors = ['#4059AD', '#97D8C4', '#F4B942', '#363636']

filename_resi = 'NND data control/well2_LTX_RESI_GFPNb_R1_400pM_2_RESI_higher_Neighbors_data1.csv'

k_nn_resi = pd.read_csv(filename_resi, sep=',',header=None)

for i in range(4):
    
    bins = np.arange(0, 1000, 2)
    ax_knn.hist(k_nn_resi[i], bins=bins, alpha=0.7, color=colors[i], edgecolor='black', linewidth=0.1, density=True)

    ax_knn.set_xlim([0, 200])
    ax_knn.set_ylim([0, 0.022])
    
    ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
    ax_knn.set_ylabel('Frequency')
    ax_knn.tick_params(direction='in')
    ax_knn.set_box_aspect(1)
    
print('Observed density = ', 1e6 * pos.shape[0]/(width*height), 'molecules per μm^2')
    