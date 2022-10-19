#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:38:40 2022

@author: masullo
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn.neighbors import NearestNeighbors

plt.close('all')


folder_exp = 'clusters RTX exp data/'
filename_exp = 'well2_CHO_A3_GFPAlfaCd20_RTXAlexa647_GFPNb_R1_300pM_1_MMStack_Pos0.ome_locs2000_RCC1000_pick48_filter13_apicked_resi_9_15_picked_multi_dbclusters_30.0_2.hdf5'
data_exp = pd.read_hdf(folder_exp + filename_exp, key = 'locs')

c_exp = '#2880C4'

px_size = 130

n_molec_exp = data_exp['n'].values
circularity_exp = data_exp['convex_circularity'].values
area_exp = data_exp['convex_hull'].values * px_size**2 

area_exp_copy = area_exp.copy()

area_exp = area_exp[area_exp_copy != 0]
circularity_exp = circularity_exp[area_exp_copy != 0]
# n_molec_exp = n_molec_exp[area_exp_copy != 0]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

bins = np.arange(0, 30, step=1) - .5
ax[0].hist(n_molec_exp, bins=bins, linewidth=1, alpha=0.5, edgecolor='k', 
           color=c_exp)

ax[0].set_xlim(0, 20)
ax[0].set_xlabel('Number of molecules in cluster')
ax[0].set_ylabel('Counts')

ax[1].hist(circularity_exp, bins=50, linewidth=1, alpha=0.5, edgecolor='k', 
           color=c_exp)

ax[1].set_xlim(0, 1)
ax[1].set_xlabel('Circularity')
ax[1].set_ylabel('Counts')

ax[2].hist(area_exp, bins=200, linewidth=1, alpha=0.5, edgecolor='k', 
           color=c_exp)

ax[2].set_xlim(0, 5000)
ax[2].set_xlabel('Area (nm^2)')
ax[2].set_ylabel('Counts')

plt.tight_layout()
