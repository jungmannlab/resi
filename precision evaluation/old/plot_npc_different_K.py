# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:26:47 2022

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

all_locs_x = np.load("all_locs_x_K5.npy")
all_locs_y = np.load("all_locs_y_K5.npy")
all_locs_z = np.load("all_locs_z_K5.npy")

subset_locs_x = all_locs_x[(all_locs_x > 28000) & (all_locs_x < 30000)]
subset_locs_y = all_locs_y[(all_locs_x > 28000) & (all_locs_x < 30000)]


fig0, ax0 = plt.subplots()
ax0.scatter(all_locs_x, all_locs_y)

fig1, ax1 = plt.subplots()
ax1.hist2d(subset_locs_x, subset_locs_y, bins=400)

ax1.set_aspect('equal')

