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

c_exp = '#388697'

px_size = 130

n_molec_exp = data_exp['n'].values
circularity_exp = data_exp['convex_circularity'].values
area_exp = data_exp['convex_hull'].values * px_size**2 

area_exp_copy = area_exp.copy()

area_exp = area_exp[area_exp_copy != 0]

# circularity_exp = circularity_exp[n_molec_exp > 3]


circularity_exp = circularity_exp[area_exp_copy != 0]
# n_molec_exp = n_molec_exp[area_exp_copy != 0]


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 4))

bins = np.arange(0, 30, step=1) - .5
ax[0, 0].hist(n_molec_exp, bins=bins, linewidth=.6, alpha=0.5, edgecolor='k', 
           color=c_exp, density=True, label='Data')

ax[0, 0].set_xlim(0, 20)
ax[0, 0].set_xlabel('Number of molecules in cluster')
ax[0, 0].set_ylabel('Norm. frequency')


ax[0, 0].tick_params(direction='in')

ax[0, 1].hist(circularity_exp, bins=np.arange(0, 1, .02), linewidth=.6, alpha=0.5, edgecolor='k', 
           color=c_exp, density=True, label='Data')

ax[0, 1].tick_params(direction='in', width=1.2)

ax[0, 1].set_xlim(0, 1)
ax[0, 1].set_xlabel('Circularity')
ax[0, 1].set_ylabel('Norm. frequency')

ax[0, 2].hist(area_exp, np.arange(0, 5000, 50), linewidth=.6, alpha=0.5, edgecolor='k', 
           color=c_exp, density=True, label='Data')


ax[0, 2].tick_params(direction='in')

ax[0, 2].set_xlim(0, 2000)
ax[0, 2].set_xlabel('Area (nm$^2$)')
ax[0, 2].set_ylabel('Norm. frequency')


folder_sim = 'clusters RTX sim data high res/'
filename_sim = 'simulated_hexamers_width_200000.0_dbclusters_32.5_2.hdf5'
data_sim = pd.read_hdf(folder_sim + filename_sim, key = 'locs')

c_sim = '#005E7C'

px_size = 130

n_molec_sim = data_sim['n'].values
circularity_sim = data_sim['convex_circularity'].values
area_sim = data_sim['convex_hull'].values * px_size**2 

area_sim_copy = area_sim.copy()

area_sim = area_sim[area_sim_copy != 0]

circularity_sim = circularity_sim[area_sim_copy != 0]
# n_molec_exp = n_molec_exp[area_exp_copy != 0]

# circularity_sim = circularity_sim[n_molec_sim == 4]


bins = np.arange(0, 30, step=1) - .5
counts_n, bins_edges_n, _ = ax[1, 0].hist(n_molec_sim, bins=bins, linewidth=1, alpha=0.5, edgecolor='k', 
                                          color=c_sim, density=True, label='Simulation')

bins_edges_n = (bins_edges_n[:-1] + bins_edges_n[1:]) / 2

ax[0, 0].plot(bins_edges_n[2:], counts_n[2:], 'o-', color=c_sim, label='Simulation')

ax[1, 0].set_xlim(0, 20)
ax[1, 0].set_xlabel('Number of molecules in cluster')
ax[1, 0].set_ylabel('Norm. frequency')

counts_c, bins_edges_c, _ = ax[1, 1].hist(circularity_sim, bins=200, linewidth=1, alpha=0.5, edgecolor='k', 
                                          color=c_sim, density=True)

bins_edges_c = (bins_edges_c[:-1] + bins_edges_c[1:]) / 2

ax[0, 1].plot(bins_edges_c, counts_c, '-', color=c_sim, label='Simulation')

ax[1, 1].set_xlim(0, 1)
ax[1, 1].set_xlabel('Circularity')
ax[1, 1].set_ylabel('Norm. frequency')

counts_a, bins_edges_a, _ = ax[1, 2].hist(area_sim, bins=np.arange(0, 5000, 25), linewidth=1, alpha=0.5, edgecolor='k', 
                                          color=c_sim, density=True)

bins_edges_a = (bins_edges_a[:-1] + bins_edges_a[1:]) / 2

ax[0, 2].plot(bins_edges_a, counts_a, '-', color=c_sim)


ax[1, 2].set_xlim(0, 5000)
ax[1, 2].set_xlabel('Area (nm$^2$)')
ax[1, 2].set_ylabel('Norm. frequency')

ax[0, 1].legend()
ax[0, 0].legend()

# plt.figure('perimeter')
# plt.hist(data_sim['convex_perimeter'][data_sim['convex_perimeter'] > 0], bins=np.arange(0, 10, 0.01))

# plt.figure('area')
# plt.hist(data_sim['convex_hull'][data_sim['convex_hull'] > 0], bins=np.arange(0, 1, 0.01))

# plt.figure('per sq')
# plt.hist(data_sim['convex_perimeter'][data_sim['convex_perimeter'] > 0]**2, bins=np.arange(0, 10, 0.01))

# a = 4 * np.pi * data_sim['convex_hull'][data_sim['convex_hull'] > 0]
# b = data_sim['convex_perimeter'][data_sim['convex_perimeter'] > 0]**2
# plt.figure('ratio')
# plt.hist(a/b, np.arange(0, 1, 0.001))

plt.tight_layout()


