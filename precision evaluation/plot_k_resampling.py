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

σ_dnapaint = 3.0

cov = [[σ_dnapaint**2, 0], [0, σ_dnapaint**2]] # create covariance matrix

data = np.random.multivariate_normal((0, 0), cov, 100)
data_reshaped = data.reshape(10, 10, 2)

fig, ax = plt.subplots(1, 3)

ax[0].scatter(data[:, 0], data[:, 1])
ax[0].set_xlim(-10, 10)
ax[0].set_ylim(-10, 10)
ax[0].set_box_aspect(1)
ax[0].set_xlabel('x (nm)')
ax[0].set_ylabel('y (nm)')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(data_reshaped.shape[0]):
    
    ax[1].scatter(data_reshaped[i, :, 0], data_reshaped[i, :, 1], c=colors[i], 
                  alpha=0.35)
    ax[1].set_xlim(-10, 10)
    ax[1].set_ylim(-10, 10)
    ax[1].set_box_aspect(1)
    
    ax[1].set_xlabel('x (nm)')
    ax[1].set_ylabel('y (nm)')
    
for i in range(data_reshaped.shape[0]):
    
    ax[2].scatter(data_reshaped[i, :, 0].mean(), 
                  data_reshaped[i, :, 1].mean(), c=colors[i], edgecolors='k', 
                  s=50)
    ax[2].set_xlim(-10, 10)
    ax[2].set_ylim(-10, 10)
    ax[2].set_box_aspect(1)
    
plt.tight_layout()
