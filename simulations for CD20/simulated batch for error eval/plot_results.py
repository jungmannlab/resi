#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:27:32 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

D_opt_array = np.load('D_opt_array_2.npy')
σ_label_opt_array = np.load('σ_label_opt_array_2.npy')
density_d_opt_array = np.load('density_d_opt_array_2.npy')

fig0, ax0 = plt.subplots()

bins = np.arange(12, 15, step=0.5) - .25
ax0.hist(D_opt_array, width=0.5 ,linewidth=1, alpha=0.5, edgecolor='k', bins=bins)
ax0.set_xlim(12, 15)
ax0.set_box_aspect(1)

ax0.set_xlabel('D (nm)')
ax0.set_ylabel('Counts')

ax0.text(14.0, 65, 'mean= ' + str(np.around(np.mean(D_opt_array), 2)) + ' nm')
ax0.text(14.0, 60, 'std = ' + str(np.around(np.std(D_opt_array), 2)) + ' nm')

fig1, ax1 = plt.subplots()

bins = np.arange(4, 7, step=0.5) - .25
ax1.hist(σ_label_opt_array, width=0.5, linewidth=1, alpha=0.5, edgecolor='k', bins=bins)
ax1.set_xlim(4, 7)
ax1.set_box_aspect(1)

ax1.set_xlabel('σ_label (nm)')
ax1.set_ylabel('Counts')

ax1.text(6.0, 65, 'mean= ' + str(np.around(np.mean(σ_label_opt_array), 2)) + ' nm')
ax1.text(6.0, 60, 'std = ' + str(np.around(np.std(σ_label_opt_array), 2)) + ' nm')

fig2, ax2 = plt.subplots()

bins = np.arange(40.5, 53.5, step=1)
ax2.hist(density_d_opt_array, width=1, linewidth=1, alpha=0.5, edgecolor='k', bins=bins)
ax2.set_xlim(42, 52)
ax2.set_box_aspect(1)

ax2.set_xlabel('frac_of_dimers (%)')
ax2.set_ylabel('Counts')

ax2.text(48.5, 23, 'mean= ' + str(np.around(np.mean(density_d_opt_array), 2)) + ' %')
ax2.text(48.5, 21, 'std = ' + str(np.around(np.std(density_d_opt_array), 2)) + ' %')