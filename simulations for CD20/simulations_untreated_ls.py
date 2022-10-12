#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:48:42 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy.signal as signal
from sklearn.neighbors import NearestNeighbors

PLOTS = False

plt.close('all')

# =============================================================================
# Load (and optionally plot) experimental data for comparison
# =============================================================================
 
filename = 'dataset1/well2_LTX_RESI_GFPNb_R1_400pM_2_MMStack_Pos0.ome_locs1000_RCC500_pick13_filter12_apicked.hdf5_varsD8_15.npz_RESI.npz'
 
data = dict(np.load(filename))
 
x = data['new_com_x_cluster']*130
y = data['new_com_y_cluster']*130
 
pos_exp = np.array([x, y]).T
 
 ### NN calculation ###
     
nbrs = NearestNeighbors(n_neighbors=5).fit(pos_exp) # find nearest neighbours
_distances_exp, _indices_exp = nbrs.kneighbors(pos_exp) # get distances and indices
 
colors = ['#2880C4', '#97D8C4', '#F4B942', '#363636']
 
for i in range(4):
 
    # compute histogram of nn-distance of the simulation
    
    distances_exp = _distances_exp[:, i+1] # get the first neighbour distances
    
    binsize = 4 # nm
    bins = np.arange(0, 1000, binsize)
    freq_exp, bins = np.histogram(distances_exp, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    if PLOTS:    
        
        fig_knn, ax_knn = plt.subplots(figsize=(5, 5))
    
        ax_knn.bar(bin_centers, freq_exp, alpha=0.3, color=colors[i], 
                   edgecolor='black', width=binsize)
    
        ax_knn.set_xlim([0, 200])
        ax_knn.set_ylim([0, 0.018])
        
        ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
        ax_knn.set_ylabel('Frequency')
        ax_knn.tick_params(direction='in')
        ax_knn.set_box_aspect(1)
        
        plt.tight_layout()
     
    if i == 0:
         
         freq_exp_1nn = freq_exp
         
# =============================================================================
# Simulation parameters
# =============================================================================   
    
dim_color = '#009FB7'
mon_color = '#FE4A49'

distribution = 'uniform'    
dimension = 2 # dimension of the simulation, d = 2 for 2D case, d = 3 for 3D
mult = 2 # multiplicity of the molecular assembly (e.g. mult = 2 for dimers)

# labelling correction
labelling = True
p = 0.5

width = 300e3 # width of the simulated area in nm
height = 300e3 # height of the simulated area in nm

total_real_density = 100e-6 # molecules per nm^2 BEFORE labeling efficiency

D_array_start = 10
D_array_stop = 16

σ_label_start = 2
σ_label_stop = 8

density_d_start = 39
density_d_stop = 51

D_array = np.arange(D_array_start, D_array_stop, step=0.5)
σ_label_array = np.arange(σ_label_start, σ_label_stop, step=0.5)
density_d_array = np.arange(density_d_start, density_d_stop, step=1)

sq_distance_array = np.zeros((len(D_array), len(σ_label_array), len(density_d_array)))

for i, D in enumerate(D_array):
    for j, sigma_label in enumerate(σ_label_array):
        for k, density in enumerate(density_d_array):
    
            print(i, j, k)
            
            t0 = time.time()
        
            # =============================================================================
            # experimental parameters
            # =============================================================================
            
            # independent parameters
            
            D_dimer = D # real dimer distance in nm
            density_d = density * 1e-6 # molecules per nm^2
            σ_label = sigma_label # nm
        
            # dependent parameters
            
            density_m = total_real_density - density_d # molecules per nm^2
        
            N_d = int(density_d/2 * width * height) # divided by two because N_d it's the number of centers of dimers
            N_m = int(density_m * width * height)
                
            # =============================================================================
            # simulate molecules positions and calculate distances
            # =============================================================================
            
            c_pos_dim = np.zeros((N_d, dimension)) # initialize array of central positions for dimers
            c_pos_mon = np.zeros((N_m, dimension)) # initialize array of central positions for monomers
            
            if dimension == 2:
                
                if distribution == 'uniform':
                    c_pos_dim[:, 0], c_pos_dim[:, 1] = [np.random.uniform(0, width, N_d), 
                                                        np.random.uniform(0, height, N_d)]
                    
                    c_pos_mon[:, 0], c_pos_mon[:, 1] = [np.random.uniform(0, width, N_m), 
                                                        np.random.uniform(0, height, N_m)]
                    
                else:
                    print('Please enter a valid distribution key')
              
            if PLOTS:
                
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
            
            pos_dim = np.zeros((N_d, dimension, mult)) # array containing all the info for each dimer
            
            # generate the positions of each molecule for each dimer
            pos_dim[:, :, 0] = c_pos_dim + np.array([D0*np.cos(angle), D0*np.sin(angle)]).T
            pos_dim[:, :, 1] = c_pos_dim - np.array([D1*np.cos(angle), D1*np.sin(angle)]).T
               
            if PLOTS:
                
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
                    
            pos_dim = np.concatenate((pos_dim[:, :, 0], pos_dim[:, :, 1]), axis=0) # flatten the array to get all molecules positions together
            pos_mon = c_pos_mon # for monomers the array stays the same
            
            if PLOTS:
                
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
            
            pos = np.concatenate((pos_dim, pos_mon)) # put together dimer and monomer molecules
                
            N = mult*N_d + N_m # total number of molecules before labelling
            
            if labelling:
                
                ids = np.random.choice(np.arange(N), size=int((N)*p), replace=False) # take a random subset of indexes of size N * p
                pos = pos[ids] # take only the labelled positions
                    
            if PLOTS:
            
                # this plot should output dimers taking into account labelling, molecules with black edge are the ones actually labelled
                ax2.scatter(pos[:, 0], pos[:, 1], facecolors='none', edgecolors='k')
            
            # =============================================================================
            # NN calculation
            # =============================================================================
        
            nbrs = NearestNeighbors(n_neighbors=2).fit(pos) # find nearest neighbours
            _distances, _indices = nbrs.kneighbors(pos) # get distances and indices
            
            distances = _distances[:, 1] # get the first neighbour distances
            
            bins = np.arange(0, 1000, 4)
            freq_sim_1nn, bin_edges = np.histogram(distances, bins=bins, density=True) # compute histogram of nn-distance of the simulation
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            if PLOTS:
                
                ax_knn.plot(bin_centers, freq_sim_1nn, '-o', color=colors[0], linewidth=1, 
                            label='uniform '+str(i+1)+'st-NN')
                
                ax_knn.set_xlim([0, 100])
                ax_knn.set_ylim([0, 0.022])
        
                ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
                ax_knn.set_ylabel('Frequency')
                ax_knn.tick_params(direction='in')
                ax_knn.set_box_aspect(1)
            
                plt.tight_layout()
                       
            sq_distance = np.sum((freq_exp_1nn - freq_sim_1nn)**2) # in nm^2
            sq_distance_array[i, j, k] = sq_distance

            print('LS distnace =', np.around(sq_distance, 7))
            # print('Dimer density =', density_d)
            # print('Observed density = ', 1e6 * pos.shape[0]/(width*height), 'molecules per μm^2')
            
            t1 = time.time()
            total_time = np.around(t1 - t0, 1)
            
            print("Simulation took ", total_time, " seconds")

                  
min_index = np.where(sq_distance_array == sq_distance_array.min())

D_opt = D_array[min_index[0]]
σ_opt = σ_label_array[min_index[1]]
density_d_opt = density_d_array[min_index[2]]

fig, ax = plt.subplots()
# ax.plot(density_d_array, sq_distance_array[min_index[0], min_index[1], :], '-o')

ax.plot(density_d_array, sq_distance_array[min_index[0], min_index[1], :].flatten(), '-o')

np.save('sq_distance_array.npy', sq_distance_array)

np.save('D_array.npy', D_array)
np.save('density_d_array.npy', density_d_array)
np.save('σ_label_array.npy', σ_label_array)


