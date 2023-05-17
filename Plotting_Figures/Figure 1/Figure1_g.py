#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:12:52 2022

@author: masullo
"""

import os
from os.path import dirname as up

cwd = os.getcwd()
wdir = up(up(cwd))
os.chdir(wdir)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tools

plt.close('all')

pxsize = 130 # nm
simulate = True

if simulate:

    # origami sites (built manually) 
    
    means_ch1 = np.array([[-21, 0], [-1, 0], [19, 0]])
    means_ch2 = means_ch1.copy()
    
    dx = 2.0 # in nm
    dx_array = np.array([[dx, 0], [dx, 0], [dx, 0]])
    
    means_ch2 = means_ch2 + dx_array
    sites = np.concatenate((means_ch1, means_ch2), axis=0)
    
    simulated_data_fname = os.getcwd() + '/simulated/simulated_data_fig1g.hdf5' 
    tools.simulate_data(simulated_data_fname, sites, locs_per_site=8000, 
                        σ_dnapaint=3.0) # creates simulated data file

# resample for different K
    
files = ['simulated/simulated_data_fig1g.hdf5']

# K_array = np.array([1, 5, 10, 50]) # number of localizations per subset
K_array = np.array([10, 40, 100]) # number of localizations per subset


fig0, ax0 = plt.subplots(4, 1, figsize=(16, 20)) # size matches len(K_array)

iterables = [ax0.reshape(-1), K_array]

for k, (ax, K) in enumerate(zip(*iterables)):

    resi_data, pre_resi_data = tools.get_resi_locs(files, K)

    all_locs_x = np.array(resi_data['0']['x']) # simulation already in nm
    all_locs_y = np.array(resi_data['0']['y']) # simulation already in nm
    
    binsmax = all_locs_x.max() + 10
    binsmin = all_locs_x.min() - 10

    bins = np.arange(binsmin, binsmax, 0.3)
    
    ax.hist2d(all_locs_x, all_locs_y, bins=bins, cmap='hot')
    ax.title.set_text('K = {}'.format(K))
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_aspect('equal')
    
    np.save(os.getcwd() + '/simulated/resi_locs_x_K' + str(K), all_locs_x)
    np.save(os.getcwd() + '/simulated/resi_locs_y_K' + str(K), all_locs_y)




# plt.close('all')
# plt.style.use('dark_background')

# pxsize = 130 # nm
# simulate = True

# if simulate:

#     # origami sites (built manually) 
    
#     means_ch1 = np.array([[-20, 0], [0, 0], [20, 0]])
#     means_ch2 = means_ch1.copy()
    
#     dx = 2.0 # in nm
#     dx_array = np.array([[dx, 0], [dx, 0], [dx, 0]])
    
#     means_ch2 = means_ch2 + dx_array
#     sites = np.concatenate((means_ch1, means_ch2), axis=0)
    
#     simulated_data_fname_1_2 = os.getcwd() + '/simulated/simulated_data_ch1_ch2.hdf5' 
#     tools.simulate_data(simulated_data_fname_1_2, sites, locs_per_site=200, 
#                         σ_dnapaint=2.0/np.sqrt(50)) # creates simulated data file
    
#     simulated_data_fname_1 = os.getcwd() + '/simulated/simulated_data_ch1.hdf5' 
#     tools.simulate_data(simulated_data_fname_1, means_ch1, locs_per_site=200, 
#                         σ_dnapaint=2.0/np.sqrt(50)) # creates simulated data file
    
#     simulated_data_fname_2 = os.getcwd() + '/simulated/simulated_data_ch2.hdf5' 
#     tools.simulate_data(simulated_data_fname_2, means_ch2, locs_per_site=200, 
#                         σ_dnapaint=2.0/np.sqrt(50)) # creates simulated data file

# # resample for different K
    
# file_1 = ['simulated/simulated_data_ch1.hdf5']
# file_2 = ['simulated/simulated_data_ch2.hdf5']

# file_1_2 = ['simulated/simulated_data_ch1_ch2.hdf5']

# K_array = [1]

# fig, ax = plt.subplots(4, 1, figsize=(10, 20)) # size matches len(K_array)

# for K in K_array:
    
#     resi_data_ch1, pre_resi_data_ch1 = tools.get_resi_locs(file_1, K)
    
#     all_locs_x_ch1 = np.array(resi_data_ch1['0']['x']) # simulation already in nm
#     all_locs_y_ch1 = np.array(resi_data_ch1['0']['y']) # simulation already in nm
    
#     resi_data_ch2, pre_resi_data_ch2 = tools.get_resi_locs(file_2, K)
    
#     all_locs_x_ch2 = np.array(resi_data_ch2['0']['x']) # simulation already in nm
#     all_locs_y_ch2 = np.array(resi_data_ch2['0']['y']) # simulation already in nm
    
#     resi_data_ch1_ch2, pre_resi_data_ch1_ch2 = tools.get_resi_locs(file_1_2, K) 
    
#     all_locs_x_ch1_ch2 = np.array(resi_data_ch1_ch2['0']['x']) # simulation already in nm
#     all_locs_y_ch1_ch2 = np.array(resi_data_ch1_ch2['0']['y'])
    
# bins = np.arange(-30, 30, 0.2)
    
# ax[0].hist2d(all_locs_x_ch1, all_locs_y_ch1, bins=bins, cmap='gist_heat')
# ax[1].hist2d(all_locs_x_ch2, all_locs_y_ch2, bins=bins, cmap='gist_heat')

# ax[2].hist2d(all_locs_x_ch1_ch2, all_locs_y_ch1_ch2, bins=bins, cmap='gist_heat')

# ax[0].set_xlim(-30, 30)
# ax[1].set_xlim(-30, 30)
# ax[2].set_xlim(-30, 30)


# ax[0].set_ylim(-10, 10)
# ax[1].set_ylim(-10, 10)
# ax[2].set_ylim(-10, 10)


# ax[0].set_aspect('equal')
# ax[1].set_aspect('equal')

# counts_ch1, bins_ch1, _ = np.histogram2d(all_locs_x_ch1, all_locs_y_ch1, bins=bins, density=True)
# counts1D_ch1 = np.sum(counts_ch1, axis=1)

# counts_ch2, bins_ch2, _ = np.histogram2d(all_locs_x_ch2, all_locs_y_ch2, bins=bins, density=True)
# counts1D_ch2 = np.sum(counts_ch2, axis=1)

# bins_centers_ch1 = (bins_ch1[:-1] + bins_ch1[1:])/2
# ax[2].bar(bins_centers_ch1, counts1D_ch1, width=1, edgecolor='k', color='#9EC1CF',
#           alpha=0.5, linewidth=0.1)


# bins_centers_ch2 = (bins_ch2[:-1] + bins_ch2[1:])/2
# ax[2].bar(bins_centers_ch2, counts1D_ch2, width=1, edgecolor='k', color='#9EE09E',
#           alpha=0.5, linewidth=0.1)
                
# #Define the Gaussian function

# def gaussian(x, A, x0, σ):
    
#     return A*np.exp(-(x-x0)**2/(2*σ**2))

# def multi_gauss(x, *pars):
    
    
#     offset = pars[-1]
#     offset = 0 # force offset to be 0
    
#     g1 = gaussian(x, pars[0], pars[1], pars[2])
#     g2 = gaussian(x, pars[3], pars[4], pars[5])
#     g3 = gaussian(x, pars[6], pars[7], pars[8])
    
#     return g1 + g2 + g3 + offset

# init_guess = np.array([10, -20, 2, 10, 0, 2, 10, 20, 2])  

# popt_ch1, pcov_ch1 = curve_fit(multi_gauss, bins_centers_ch1, counts1D_ch1, p0=init_guess)
# popt_ch2, pcov_ch2 = curve_fit(multi_gauss, bins_centers_ch2, counts1D_ch2, p0=init_guess)


# ax[2].plot(np.linspace(bins[0], bins[-2], 1000), 
#             multi_gauss(np.linspace(bins[0], bins[-2], 1000), *popt_ch1),
#             color='#27AAE1', linewidth=3)

# ax[2].plot(np.linspace(bins[0], bins[-2], 1000), 
#             multi_gauss(np.linspace(bins[0], bins[-2], 1000), *popt_ch2),
#             color='#9EE09E', linewidth=3)
    
# ax[2].set_xlim(-30, 30)

# popt_ch1_resi = popt_ch1

# popt_ch1_resi[2], popt_ch1_resi[5], popt_ch1_resi[8]  = popt_ch1[2]/np.sqrt(200), popt_ch1[5]/np.sqrt(200), popt_ch1[8]/np.sqrt(200)
# popt_ch1_resi[0], popt_ch1_resi[3], popt_ch1_resi[6]  = 1, 1, 1

# ax[3].plot(np.linspace(bins[0], bins[-2], 10000), multi_gauss(np.linspace(bins[0], bins[-2], 10000), *popt_ch1_resi),
#            color='#27AAE1', linewidth=3)

# popt_ch2_resi = popt_ch2

# popt_ch2_resi[2], popt_ch2_resi[5], popt_ch2_resi[8]  = popt_ch2[2]/np.sqrt(200), popt_ch2[5]/np.sqrt(200), popt_ch2[8]/np.sqrt(200)
# popt_ch2_resi[0], popt_ch2_resi[3], popt_ch2_resi[6]  = 1, 1, 1


# ax[3].plot(np.linspace(bins[0], bins[-2], 10000), multi_gauss(np.linspace(bins[0], bins[-2], 10000), *popt_ch2_resi),
#            color='#9EE09E', linewidth=3)

# ax[3].set_xlim(-30, 30)
# ax[3].set_ylim(0, 1.1)