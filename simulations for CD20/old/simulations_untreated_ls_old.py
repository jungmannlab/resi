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



plt.close('all')

density_d_array = np.array([0, 10, 20, 30, 35, 38, 40, 41, 42, 43, 44, 45, 46, 47, 50, 55, 60, 65, 70, 80, 90, 100])
# density = 43
sigma_label = 5

# σ_label_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

sq_distance_array = np.zeros(len(density_d_array))
# sq_distance_array = np.zeros(len(σ_label_array))


# density_d_array = np.array([45])
values = len(density_d_array)
# values = len(σ_label_array)

# D_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# values = len(D_array)
Dd = 13
# sq_distance_array = np.zeros(len(D_array))

# for j, Dd in enumerate(D_array):
# for j, sigma_label in enumerate(σ_label_array):
for j, density in enumerate(density_d_array):
    
    t0 = time.time()

    # =============================================================================
    # experimental parameters
    # =============================================================================
    
    # independent parameters
    
    D = 2 # dimension of the simulation, d = 2 for 2D case, d = 3 for 3D
    mult = 2 # multiplicity of the molecular assembly (e.g. mult = 2 for dimers)
    
    D_dimer = Dd # real dimer distance in nm
    total_real_density = 100e-6 # molecules per nm^2 BEFORE labeling efficiency
    density_d = density * 1e-6 # molecules per nm^2
    density_m = total_real_density - density_d # molecules per nm^2
    
    σ_label = sigma_label # nm
    width = 400e3 # width of the simulated area in nm
    height = 400e3 # height of the simulated area in nm
    
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
    
    plot_examples = False
    
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
    
    np.random.shuffle(pos) # shuffle array in order not to have first dimers and then monomers
    
    N = mult*N_d + N_m # total number of molecules before labelling
    
    if labelling:
        
        ids = np.random.choice(np.arange(N), size=int((N)*p), replace=False) # take a random subset of indexes of size N * p
        pos = pos[ids] # take only the labelled positions
        
    print(pos.shape)
    
    if plot_examples == True:
    
        # this plot should output dimers taking into account labelling, molecules with black edge are the ones actually labelled
        ax2.scatter(pos[:, 0], pos[:, 1], facecolors='none', edgecolors='k')
    
    ### NN calculation ###
        
    nbrs = NearestNeighbors(n_neighbors=5).fit(pos) # find nearest neighbours
    _distances, _indices = nbrs.kneighbors(pos) # get distances and indices
    # distances = _distances[:, 1] # get the first neighbour distances
    
    colors = ['#2880C4', '#97D8C4', '#F4B942', '#363636']
    
    if values == 1:
    
        fig_knn, ax_knn = plt.subplots(figsize=(5, 5))
    
    for i in range(4):
    
        # plot histogram of nn-distance of the simulation
        
        distances = _distances[:, i+1] # get the first neighbour distances
        
        bins = np.arange(0, 1000, 4)
        freq_sim, bins = np.histogram(distances, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
    
    
        if values == 1:
            
            ax_knn.plot(bin_centers, freq_sim, '-o', color=colors[i], linewidth=1, 
                        label='uniform '+str(i+1)+'st-NN')
            
            ax_knn.set_xlim([0, 100])
            ax_knn.set_ylim([0, 0.022])
    
            ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
            ax_knn.set_ylabel('Frequency')
            ax_knn.tick_params(direction='in')
            ax_knn.set_box_aspect(1)
        
            plt.tight_layout()
        
        if i == 0:
            
            freq_sim_1nn = freq_sim
        
    # =============================================================================
    # plot experimental data for comparison
    # =============================================================================
    
    filename = 'dataset1/well2_LTX_RESI_GFPNb_R1_400pM_2_MMStack_Pos0.ome_locs1000_RCC500_pick13_filter12_apicked.hdf5_varsD8_15.npz_RESI.npz'
    
    data = dict(np.load(filename))
    
    x = data['new_com_x_cluster']*130
    y = data['new_com_y_cluster']*130
    
    pos_exp = np.array([x, y]).T
    
    from sklearn.neighbors import NearestNeighbors
    
    ### NN calculation ###
        
    nbrs = NearestNeighbors(n_neighbors=5).fit(pos_exp) # find nearest neighbours
    _distances_exp, _indices_exp = nbrs.kneighbors(pos_exp) # get distances and indices
    # distances = _distances[:, 1] # get the first neighbour distances
    
    colors = ['#2880C4', '#97D8C4', '#F4B942', '#363636']
    
    for i in range(4):
    
        # plot histogram of nn-distance of the simulation
        
        distances_exp = _distances_exp[:, i+1] # get the first neighbour distances
        
        binsize = 4 # nm
        bins = np.arange(0, 1000, binsize)
        freq_exp, bins = np.histogram(distances_exp, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
    
        if values == 1:    
    
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
        
    sq_distance = np.sum((freq_exp_1nn - freq_sim_1nn)**2) # in nm^2
    
    sq_distance_array[j] = sq_distance
    
    print('Dimer density =', density_d)
    print('LS distnace =', np.around(sq_distance, 7))
        
    print('Observed density = ', 1e6 * pos.shape[0]/(width*height), 'molecules per μm^2')
    
    
    # =============================================================================
    # plot 1st NN and compared to only monomers distribution
    # =============================================================================
        
    # colors = ['#2D7DD2']
    # fig_1stnn, ax_1stnn = plt.subplots(figsize=(5, 5))
    
    # for i in range(1):
    
    #     # plot histogram of nn-distance of the simulation
        
    #     distances = _distances[:, i+1] # get the first neighbour distances
        
    #     freq, bins = np.histogram(distances, bins=200, density=True)
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
        
    #     bins = np.arange(0, 1000, 4)
    #     ax_1stnn.hist(_distances_exp[:, i+1], bins=bins, alpha=0.5, 
    #                   color='#2880C4', edgecolor='black', linewidth=0.1, 
    #                   density=True)
        
    #     ax_1stnn.plot(bin_centers, freq, color=colors[i], linewidth=2, 
    #                 label='uniform '+str(i+1)+'st-NN')
        
    #     plt.tight_layout()
        
        
    # ax_1stnn.set_xlim([0, 100])
    # ax_1stnn.set_ylim([0, 0.022])
    
    # ax_1stnn.set_xlabel('K-th nearest-neighbour distance (nm)')
    # ax_1stnn.set_ylabel('Frequency')
    # ax_1stnn.tick_params(direction='in')
    # ax_1stnn.set_box_aspect(1)
    
    t1 = time.time()
    
    total_time = np.around(t1 - t0, 1)
    
    print("Simulation took ", total_time, " seconds")
    

fig, ax = plt.subplots()

ax.plot(density_d_array, sq_distance_array, '-o')