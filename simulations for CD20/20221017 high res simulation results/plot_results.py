#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:36:17 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt

D_array = np.load('D_array.npy')
σ_label_array = np.load('σ_label_array.npy')
density_d_array = np.load('density_d_array.npy')

sq_distance_array = np.load('sq_distance_array.npy')

min_index = np.where(sq_distance_array == sq_distance_array.min())
min_index = np.array(min_index).flatten()

D_opt = D_array[min_index[0]]
σ_opt = σ_label_array[min_index[1]]
density_d_opt = density_d_array[min_index[2]]

print(D_opt, σ_opt, density_d_opt)

fig0, ax0 = plt.subplots()
ax0.plot(D_array, sq_distance_array[:, min_index[1], min_index[2]])

fig1, ax1 = plt.subplots()
ax1.plot(σ_label_array, sq_distance_array[min_index[0], :, min_index[2]])

fig2, ax2 = plt.subplots()
ax2.plot(density_d_array, sq_distance_array[min_index[0], min_index[1], :])