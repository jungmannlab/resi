#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 23:37:41 2022

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

# =============================================================================
# real data
# =============================================================================

path = r'/Volumes/pool-miblab4/users/masullo/z.microscopy_processed/resi origami - K analysis/20 nm origami'

i = 17

files = [path + '/R1_R4_apicked_ori' + str(i) + '_ClusterD5_40.hdf5', 
          path + '/R3_R4_apicked_ori' + str(i) + '_aligned_ClusterD5_40.hdf5']

# ori = 42

# files = ['test_files/5nm/R1_R4_ori'+str(ori)+'_ClusterD4_50.hdf5', 
#           'test_files/5nm/R3_R4_ori'+str(ori)+'_aligned_ClusterD4_50.hdf5']

K_array = np.array([1, 40]) # number of localizations per subset
fig0, ax0 = plt.subplots(1, 2, figsize=(10, 8)) # size must match

for k, (ax, K) in enumerate(zip(ax0.reshape(-1), K_array)):

    # pre_resi_data are the dna-paint localizations with a (random) resampling
    # label but not having averaged anything yet
    resi_data, pre_resi_data = tools.get_resi_locs(files, K)
    
    all_locs_x = np.append(np.array(resi_data['0']['x']),
                                np.array(resi_data['1']['x'])) * pxsize
    
    all_locs_y = np.append(np.array(resi_data['0']['y']),
                           np.array(resi_data['1']['y'])) * pxsize
        
    binsx = np.linspace(all_locs_x.min()-10, all_locs_x.max()+10, 300)
    binsy = np.linspace(all_locs_y.min()-10, all_locs_y.max()+10, 300)
    
    ax.hist2d(all_locs_x, all_locs_y, bins=[binsx, binsy], cmap='hot')
    ax.title.set_text('K = '+str(K))
    ax.set_aspect('equal')
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    
plt.tight_layout()
