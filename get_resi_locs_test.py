#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 19:21:19 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tools

plt.close('all')

pxsize = 130 # nm
K = 20

files = ['test_files/R1_picked_ori90_ClusterD4_50.hdf5', 
         'test_files/R3_picked_ori90_aligned_ClusterD4_50.hdf5']

resi_data, _ = tools.get_resi_locs(files, K)

all_locs_x = np.append(np.array(resi_data['0']['x']),
                            np.array(resi_data['1']['x'])) * pxsize

all_locs_y = np.append(np.array(resi_data['0']['y']),
                       np.array(resi_data['1']['y'])) * pxsize

fig2, ax2 = plt.subplots()

ax2.hist2d(all_locs_x, all_locs_y, bins=300, cmap='hot')
ax2.set_xlabel('x (nm)')
ax2.set_ylabel('y (nm)')
ax2.set_aspect('equal')