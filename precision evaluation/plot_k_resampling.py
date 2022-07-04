#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 22:14:20 2022

@author: masullo
"""

import os
from os.path import dirname as up

cwd = os.getcwd()
wdir = up(cwd)
os.chdir(wdir)

import numpy as np
import matplotlib.pyplot as plt
import tools

plt.close('all')
plt.rcParams.update({'font.size': 18})

σ_dnapaint = 3.0

cov = [[σ_dnapaint**2, 0], [0, σ_dnapaint**2]] # create covariance matrix

data = np.random.multivariate_normal((12, 0), cov, 100)
data_reshaped = data.reshape(10, 10, 2)


# origami sites (built manually) 

means_ch1 = np.array([[-50, -7.5], [-50, 7.5], [50, 0], 
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

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].scatter(sites[:, 0], sites[:, 1], facecolors='grey', edgecolors='k', s=50, marker='*')
ax[0, 0].set_box_aspect(1)
ax[0, 0].set_xlabel('x (nm)')
ax[0, 0].set_ylabel('y (nm)')
ax[0, 0].set_xlim(-30, 30)
ax[0, 0].set_ylim(-30, 30)

ax[0, 1].scatter(data[:, 0], data[:, 1], s=100)
ax[0, 1].set_xlim(1, 23)
ax[0, 1].set_ylim(-11, 11)
ax[0, 1].set_box_aspect(1)
ax[0, 1].set_xlabel('x (nm)')
ax[0, 1].set_ylabel('y (nm)')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(data_reshaped.shape[0]):
    
    if i == 0:
        
            ax[1, 0].scatter(data_reshaped[i, :, 0], data_reshaped[i, :, 1], 
                             c=colors[i], 
                             alpha=1, edgecolors='k', s=100)
        
    else:
    
        ax[1, 0].scatter(data_reshaped[i, :, 0], data_reshaped[i, :, 1], 
                         c=colors[i], 
                         alpha=0.35, s=100)

ax[1, 0].set_xlim(1, 23)
ax[1, 0].set_ylim(-11, 11)
ax[1, 0].set_box_aspect(1)

ax[1, 0].set_xlabel('x (nm)')
ax[1, 0].set_ylabel('y (nm)')
    
for i in range(data_reshaped.shape[0]):
    
    if i == 0:
        
        ax[1, 1].scatter(data_reshaped[i, :, 0].mean(), 
                      data_reshaped[i, :, 1].mean(), 
                      c=colors[i], edgecolors='k', s=100)
        
    else:
    
        ax[1, 1].scatter(data_reshaped[i, :, 0].mean(), 
                         data_reshaped[i, :, 1].mean(), c=colors[i], 
                         s=100, alpha=0.35)
        
ax[1, 1].set_xlim(1, 23)
ax[1, 1].set_ylim(-11, 11)
ax[1, 1].set_box_aspect(1)

ax[1, 1].set_xlabel('x (nm)')
ax[1, 1].set_ylabel('y (nm)')

plt.tight_layout()
