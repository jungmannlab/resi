#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:48:42 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.close('all')

# =============================================================================
# experimental parameters
# =============================================================================

# independent parameters

D = 2 # dimension of the simulation, d = 2 for 2D case, d = 3 for 3D
mult = 2 # multiplicity of the molecular assembly (e.g. mult = 2 for dimers)

D_dimer = 12 # real dimer distance in nm
density_d = 154e-6 # molecules per nm^2 (or nm^3)
density_m = 154e-6 # molecules per nm^2 (or nm^3)

σ_label = 7 # nm
width = 40e3 # width of the simulated area in nm
height = 40e3 # height of the simulated area in nm
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
N_d = int(density_d/2 * width * height) # divided by two because N_d it's the number of centers of dimers
N_m = int(density_m * width * height)

plot_examples = True

# =============================================================================
# simulate molecules positions and calculate distances
# =============================================================================

c_pos_dim = np.zeros((N_d, D)) # initialize array of central positions for dimers
c_pos_mon = np.zeros((N_m, D)) # initialize array of central positions for monomers

if D == 2:
    
    if distribution == 'uniform':
        c_pos_dim[:, 0], c_pos_dim[:, 1] = [np.random.uniform(0, width, N_d), 
                                            np.random.uniform(0, height, N_d)]
        
        c_pos_mon[:, 0], c_pos_mon[:, 1] = [np.random.uniform(0, width, N_m), 
                                            np.random.uniform(0, height, N_m)]
        
    else:
        print('Please enter a valid distribution key')
        
if D == 3:
    
    if distribution == 'uniform':
        
        c_pos_dim[:, 0] = np.random.uniform(0, width, N_d) 
        c_pos_dim[:, 1] = np.random.uniform(0, height, N_d)
        c_pos_dim[:, 2] = np.random.uniform(0, depth, N_d)
        
        c_pos_mon[:, 0] = np.random.uniform(0, width, N_m) 
        c_pos_mon[:, 1] = np.random.uniform(0, height, N_m)
        c_pos_mon[:, 2] = np.random.uniform(0, depth, N_m)

    else:
        print('Please enter a valid distribution key')
  
if plot_examples:
    
    if D == 2:
    
        fig0, ax0 = plt.subplots() # dimers
        fig0.suptitle('Dimers + their center')
        fig1, ax1 = plt.subplots() # monomers
        fig1.suptitle('Monomers')
                  
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


# generate angles and distances 0 and 1 for each dimer, total N_d dimers

if D == 2:
    
    angle = np.random.uniform(0, 2*np.pi, N_d) # generate random dimer orientations
    D0 = np.random.normal(loc=D_dimer/2, scale=σ_label, size=N_d) # distances of molecule 0 to the dimer center
    D1 = np.random.normal(loc=D_dimer/2, scale=σ_label, size=N_d) # distances of molecule 1 to the dimer center
    
    pos_dim = np.zeros((N_d, D, mult)) # array containing all the info for each dimer
    
    # generate the positions of each molecule for each dimer
    pos_dim[:, :, 0] = c_pos_dim + np.array([D0*np.cos(angle), D0*np.sin(angle)]).T
    pos_dim[:, :, 1] = c_pos_dim - np.array([D1*np.cos(angle), D1*np.sin(angle)]).T
    
elif D == 3:
    
    pass

    #TODO: write random sample in shperical coordinates


if plot_examples:
    
    if D == 2:
    
        # this plot should output dimers with its center, and two molecules marked with the same color
        ax0.scatter(pos_dim[:, :, 0][:, 0], pos_dim[:, :, 0][:, 1], alpha=0.5, 
                    color=dim_color )
        ax0.scatter(pos_dim[:, :, 1][:, 0], pos_dim[:, :, 1][:, 1], alpha=0.5, 
                    color=dim_color )
        
        length = 1000 # nm, length of the display area for the graph
        
        ax0.set_xlim(width/2, width/2 + length)
        ax0.set_ylim(width/2, width/2 + length)
        
        ax1.set_xlim(width/2, width/2 + length)
        ax1.set_ylim(width/2, width/2 + length)
        
# =============================================================================
# Labeling efficiency
# =============================================================================

# flatten the array to get all molecules positions together
pos_dim = np.concatenate((pos_dim[:, :, 0], pos_dim[:, :, 1]), axis=0) 
pos_mon = c_pos_mon # for monomers the array stays the same

if plot_examples:
    
    # this plot should output dimers without its center, 
    # two molecules marked with the same color AND the monomers in another color
    fig2, ax2 = plt.subplots()
    fig2.suptitle('Monomers + dimers')
    
    ax2.scatter(pos_dim[:, 0], pos_dim[:, 1], alpha=0.25, color=dim_color)
    ax2.scatter(pos_mon[:, 0], pos_mon[:, 1], alpha=0.25, color=mon_color)
    
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    ax2.set_title('Real density = '+str(int((density_d + density_m)*1e6))+'/$μm^2$')
    ax2.set_box_aspect(1)
    
    ax2.set_xlim(width/2, width/2 + length)
    ax2.set_ylim(width/2, width/2 + length)
    

# put together dimer and monomer molecules
pos = np.concatenate((pos_dim, pos_mon)) 

np.random.shuffle(pos) # shuffle array in order not to have first dimers and then monomers

N = mult*N_d + N_m # total number of molecules before labelling

if labelling:
    
    ids = np.random.choice(np.arange(N), size=int((N)*p), replace=False) # take a random subset of indexes of size N * p
    pos = pos[ids] # take only the labelled positions
    
print(pos.shape)

# this plot should output dimers taking into account labelling, molecules with black edge are the ones actually labelled
ax2.scatter(pos[:, 0], pos[:, 1], facecolors='none', edgecolors='k')

# this plot should output dimers taking into account labelling and identity lost, just like data looks like

fig3, ax3 = plt.subplots()
ax3.set(facecolor='black')
fig3.suptitle('Monomers + dimers not distinguishable')

ax3.scatter(pos[:, 0], pos[:, 1], facecolors='orange', edgecolors='none', s=10)

ax3.set_xlabel('x (nm)')
ax3.set_ylabel('y (nm)')
ax3.set_title('Real density = '+str(int((density_d + density_m)*1e6))+'/$μm^2$')
ax3.set_box_aspect(1)

length_ax3 = 2000
ax3.set_xlim(width/2, width/2 + length_ax3)
ax3.set_ylim(width/2, width/2 + length_ax3)

# =============================================================================
# Create and apply mask
# =============================================================================

# create 2D hist (image) from the coordinates of the molecules (pos)
x0_hist = 20000
y0_hist = 33000
length_hist = 12000
binsize = 20 # in nm
bins_x = np.arange(x0_hist, x0_hist + length_hist, binsize)
bins_y = np.arange(y0_hist, y0_hist + length_hist, binsize)
counts, xedges, yedges, *_ = ax1.hist2d(x, y, bins=[bins_x, bins_y], cmap='hot')

# =============================================================================
# NN calculation
# =============================================================================

from sklearn.neighbors import NearestNeighbors

### NN calculation ###
    
nbrs = NearestNeighbors(n_neighbors=5).fit(pos) # find nearest neighbours
_distances, _indices = nbrs.kneighbors(pos) # get distances and indices
# distances = _distances[:, 1] # get the first neighbour distances

colors = ['#2880C4', '#97D8C4', '#F4B942', '#363636']
fig_knn, ax_knn = plt.subplots(figsize=(5, 5))

for i in range(4):

    # plot histogram of nn-distance of the simulation
    
    distances = _distances[:, i+1] # get the first neighbour distances
    
    freq, bins = np.histogram(distances, bins=200, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # if i == 0:
    
    #     data_aux = np.zeros((2, len(freq)))
    #     data_aux[0, :] = bin_centers
    #     data_aux[1, :] = freq
    #     np.save('data_only_mon.npy', data_aux)

    ax_knn.plot(bin_centers, freq, color=colors[i], linewidth=2, 
                label='uniform '+str(i+1)+'st-NN')
    
    plt.tight_layout()
    
ax_knn.set_xlim([0, 100])
ax_knn.set_ylim([0, 0.022])

ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
ax_knn.set_ylabel('Frequency')
ax_knn.tick_params(direction='in')
ax_knn.set_box_aspect(1)

# =============================================================================
# Plot experimental data for comparison
# =============================================================================

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
    
    # freq, bins = np.histogram(distances, bins=100, density=True)
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # ax_knn.plot(bin_centers, freq, color=colors[i], linewidth=2, 
    #             label='uniform '+str(i+1)+'st-NN')
    
    bins = np.arange(0, 1000, 4)
    ax_knn.hist(distances_exp, bins=bins, alpha=0.5, color=colors[i], edgecolor='black', linewidth=0.1, density=True)

    ax_knn.set_xlim([0, 200])
    ax_knn.set_ylim([0, 0.025])
    
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

# filename_resi = 'dataset1/well2_LTX_RESI_GFPNb_R1_400pM_2_RESI_higher_Neighbors_data1.csv'

# # filename_resi = 'dataset2/well2_neg_GFPNb_R1_300pM_1_MMStack_Pos4.ome_locs_RCC1000_pick10_aligned_apicked_filter12_varsD9_25.npz_RESI.npz_higher_Neighbors_data1.csv'

# k_nn_resi = pd.read_csv(filename_resi, sep=',',header=None)

# for i in range(4):
    
#     bins = np.arange(0, 1000, 1)
#     ax_knn.hist(k_nn_resi[i], bins=bins, alpha=0.7, color=colors[i], edgecolor='black', linewidth=0.1, density=True)

#     ax_knn.set_xlim([0, 200])
#     ax_knn.set_ylim([0, 0.022])
    
#     ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
#     ax_knn.set_ylabel('Frequency')
#     ax_knn.tick_params(direction='in')
#     ax_knn.set_box_aspect(1)
    
print('Observed density = ', 1e6 * pos.shape[0]/(width*height), 'molecules per μm^2')


# =============================================================================
# plot 1st NN and compared to only monomers distribution
# =============================================================================
    
colors = ['#2D7DD2']
fig_1stnn, ax_1stnn = plt.subplots(figsize=(5, 5))

for i in range(1):

    # plot histogram of nn-distance of the simulation
    
    distances = _distances[:, i+1] # get the first neighbour distances
    
    freq, bins = np.histogram(distances, bins=200, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bins = np.arange(0, 1000, 4)
    ax_1stnn.hist(_distances_exp[:, i+1], bins=bins, alpha=0.5, 
                  color='#2880C4', edgecolor='black', linewidth=0.1, 
                  density=True)
    
    ax_1stnn.plot(bin_centers, freq, color=colors[i], linewidth=2, 
                label='uniform '+str(i+1)+'st-NN')
    
    plt.tight_layout()
    
    
ax_1stnn.set_xlim([0, 100])
ax_1stnn.set_ylim([0, 0.022])

ax_1stnn.set_xlabel('K-th nearest-neighbour distance (nm)')
ax_1stnn.set_ylabel('Frequency')
ax_1stnn.tick_params(direction='in')
ax_1stnn.set_box_aspect(1)