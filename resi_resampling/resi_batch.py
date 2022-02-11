#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:12:52 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tools
import os

plt.close('all')

path = r'/Volumes/pool-miblab4/users/masullo/z.microscopy_processed/resi origami - K analysis/test_data_2'

pxsize = 130 # nm
K_array = np.array([1, 5, 10, 20, 30, 40]) # number of localizations per subset
noris = 92

for i in range(noris):
    
    files = [path + '/R1_picked_ori' + str(i) + '_ClusterD4_50.hdf5', 
             path + '/R3_picked_ori' + str(i) + '_aligned_ClusterD4_50.hdf5']

    for k, K in enumerate(K_array):
        
        resi_data, _ = tools.get_resi_locs(files, K)
        
        dataframes = [resi_data['0'], resi_data['1']]
        merged_resi = pd.concat(dataframes)
        
        tools.picasso_hdf5(merged_resi, '/resi_results/resi_ori' + str(i) + '_K' + str(K) + '_.hdf5', 
                           '/R1_picked_ori' + str(i) + '_ClusterD4_50.hdf5', path)
        
        # loop over the whole folder and create files