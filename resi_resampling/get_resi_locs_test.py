#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 19:21:19 2022

@author: masullo
"""

import os
from os.path import dirname as up

cwd = os.getcwd()
wdir = up(cwd)
os.chdir(wdir)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tools

plt.close('all')

pxsize = 130 # nm
K = 20

# =============================================================================
# real data
# =============================================================================

files = ['test_files/R1_picked_ori90_ClusterD4_50.hdf5', 
          'test_files/R3_picked_ori90_aligned_ClusterD4_50.hdf5']

# pre_resi_data are the dna-paint localizations with a (random) resampling
# label but not having averaged anything yet
resi_data, pre_resi_data = tools.get_resi_locs(files, K)

all_locs_x = np.append(np.array(resi_data['0']['x']),
                            np.array(resi_data['1']['x'])) * pxsize

all_locs_y = np.append(np.array(resi_data['0']['y']),
                       np.array(resi_data['1']['y'])) * pxsize

fig2, ax2 = plt.subplots()

ax2.hist2d(all_locs_x, all_locs_y, bins=300, cmap='hot')
ax2.set_xlabel('x (nm)')
ax2.set_ylabel('y (nm)')
ax2.set_aspect('equal')


# =============================================================================
# simulated data
# =============================================================================

# origami sites (built manually) 

means_ch1 = np.array([[-30, -7.5], [-30, 7.5], [30, 0], 
                      [40, 0], [40, 20], [40, -15],
                      [-10, 20], [-10, 0], [-10, -20],
                      [10, 20], [10, 0], [10, -20]])

means_ch2 = means_ch1.copy()

dx = 2.0 # in nm

dx_array = np.array([[0, 0], [0, 0], [0, 0], 
                     [0, 0], [0, 0], [0, 0],
                     [dx, 0], [dx, 0], [dx, 0],
                     [dx, 0], [dx, 0], [dx, 0]])

means_ch2 = means_ch2 + dx_array
sites = np.concatenate((means_ch1, means_ch2), axis=0)
filename = os.getcwd() + '/simulated/simulated_data.hdf5'

# simulate localizations data

tools.simulate_data(filename, sites, locs_per_site=300, σ_dnapaint=2.0)

files = ['simulated/simulated_data.hdf5']

resi_sim_data, _ = tools.get_resi_locs(files, K)

all_locs_x = np.array(resi_sim_data['0']['x']) # simulation already in nm
all_locs_y = np.array(resi_sim_data['0']['y']) # simulation already in nm

fig3, ax3 = plt.subplots()

binsmax = all_locs_x.max() + 10
binsmin = all_locs_x.min() - 10

bins = np.arange(binsmin, binsmax, 0.2)

ax3.hist2d(all_locs_x, all_locs_y, bins=bins, cmap='hot')
ax3.set_xlabel('x (nm)')
ax3.set_ylabel('y (nm)')
ax3.set_aspect('equal')