#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:12:52 2022

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

path = r'/Volumes/pool-miblab4/users/masullo/z.microscopy_processed/resi origami - K analysis/analysis'

pxsize = 130 # nm
K_array = np.array([1, 5, 10, 20, 30, 40]) # number of localizations per subset
noris = 89

for i in range(noris):
    
    print(i)
    
    files = [path + '/R1_R4_ori' + str(i) + '_ClusterD4_50.hdf5', 
             path + '/R3_R4_ori' + str(i) + '_aligned_ClusterD4_50.hdf5']

    for k, K in enumerate(K_array):
        
        resi_data, _ = tools.get_resi_locs(files, K)
        
        dataframes = [resi_data['0'], resi_data['1']]
        merged_resi = pd.concat(dataframes)
        
        tools.picasso_hdf5(merged_resi, '/resi_results/resi_ori' + str(i) + '_K' + str(K) + '_.hdf5', 
                           '/R1_R4_ori' + str(i) + '_ClusterD4_50.hdf5', path)
        
        # loop over the whole folder and create files
