#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:25:42 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import configparser
from datetime import datetime

plt.close('all')

# =============================================================================
# load high res sim data
# =============================================================================

freq_hr = np.load('sim_data_high_res/sim_high_res_freq.npy')
bin_centers_hr = np.load('sim_data_high_res/sim_high_res_bin_centers.npy')

# =============================================================================
# experimental parameters
# =============================================================================

nnd_filename = 'CD20_1stnn_sim'

# independent parameters

D = 2 # dimension of the simulation, d = 2 for 2D case, d = 3 for 3D
mult = 2 # multiplicity of the molecular assembly (e.g. mult = 2 for dimers)

D_dimer = 13.5 # real dimer distance in nm
density_d = 47e-6 # molecules per nm^2 (or nm^3)
density_m = 53e-6 # molecules per nm^2 (or nm^3)

σ_label = 5.5 # nm
width = 22e3 # width of the simulated area in nm
height = 20e3 # height of the simulated area in nm
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


for j in range(1):
    
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
      
    if plot_examples:
        
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
    angle = np.random.uniform(0, 2*np.pi, N_d) # generate random dimer orientations
    D0 = np.random.normal(loc=D_dimer/2, scale=σ_label, size=N_d) # distances of molecule 0 to the dimer center
    D1 = np.random.normal(loc=D_dimer/2, scale=σ_label, size=N_d) # distances of molecule 1 to the dimer center
    
    pos_dim = np.zeros((N_d, D, mult)) # array containing all the info for each dimer
    
    # generate the positions of each molecule for each dimer
    pos_dim[:, :, 0] = c_pos_dim + np.array([D0*np.cos(angle), D0*np.sin(angle)]).T
    pos_dim[:, :, 1] = c_pos_dim - np.array([D1*np.cos(angle), D1*np.sin(angle)]).T
    
    
    if plot_examples:
        
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
    # NN calculation
    # =============================================================================
    
    from sklearn.neighbors import NearestNeighbors
    
    # flatten the array to get all molecules positions together
    pos_dim = np.concatenate((pos_dim[:, :, 0], pos_dim[:, :, 1]), axis=0) 
    pos_mon = c_pos_mon # for monomers the array stays the same
    
    if plot_examples:
        
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
    
    # put together dimer and monomer molecules
    pos = np.concatenate((pos_dim, pos_mon)) 
    
    # np.random.shuffle(pos) # shuffle array in order not to have first dimers and then monomers (should be irrelevant)
    
    N = mult*N_d + N_m # total number of molecules before labelling
    
    if labelling:
        
        ids = np.random.choice(np.arange(N), size=int((N)*p), replace=False) # take a random subset of indexes of size N * p
        pos = pos[ids] # take only the labelled positions
        
    print(pos.shape)
    
    # this plot should output dimers taking into account labelling, molecules with black edge are the ones actually labelled
    # ax2.scatter(pos[:, 0], pos[:, 1], facecolors='none', edgecolors='k')
    
    ### NN calculation ###
        
    nbrs = NearestNeighbors(n_neighbors=5).fit(pos) # find nearest neighbours
    _distances, _indices = nbrs.kneighbors(pos) # get distances and indices
    # distances = _distances[:, 1] # get the first neighbour distances
    
    colors = ['#2880C4', '#97D8C4', '#F4B942', '#363636']
    fig_knn, ax_knn = plt.subplots(figsize=(5, 5))
    
    for i in range(1):
    
        # plot histogram of nn-distance of the simulation
        
        distances = _distances[:, i+1] # get the first neighbour distances
        
        bins = np.arange(0, 1000, 4)
        freq_sim_1nn, bin_edges = np.histogram(distances, bins=bins, density=True) # compute histogram of nn-distance of the simulation
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ax_knn.bar(bin_centers, freq_sim_1nn, color=colors[i], edgecolor='k', linewidth=1, width=4, 
                    label='uniform '+str(i+1)+'st-NN', alpha=0.2)
        
        plt.tight_layout()
        
    ax_knn.set_xlim([0, 100])
    ax_knn.set_ylim([0, 0.022])
    
    ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
    ax_knn.set_ylabel('Frequency')
    ax_knn.tick_params(direction='in')
    ax_knn.set_box_aspect(1)
    
    ax_knn.plot(bin_centers_hr, freq_hr, color=colors[i], linewidth=2, 
                label='uniform '+str(i+1)+'st-NN')
    
    ### Save the simulated distances
    
    np.save(nnd_filename + str(j) + '.npy', distances)

### Create config file with parameters

config = configparser.ConfigParser()

config['params'] = {

'Date and time': str(datetime.now()),
'D': D,
'mult': mult,
'd_dimer (nm)': D_dimer,
'density_d (nm^-2)': density_d,
'density_m (nm^-2)': density_m,
'σ_label (nm)': σ_label,
'width (nm)': width,
'height (nm)': height,
'distribution': distribution,
'labeling efficiency': p
}

with open(nnd_filename + '_params.txt', 'w') as configfile:
    config.write(configfile)

# generate M simulated histograms and save in separate folder + metadata