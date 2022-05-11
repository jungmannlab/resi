#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:21:22 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
import skimage.filters as filters

plt.close('all')

locs = {}
method = 'RESI'

for i in range(1,14):
    
    locs[str(i)] = pd.read_csv(method + '_030522' + '/' + 'avg_locs_'+str(i)+'.csv')
    

data = np.array(locs['12'])

data_top = data[data[:, 2] > 0]
data_bottom = data[data[:, 2] < 0]
nbins = 1400
xbins = np.linspace(-70, 70, nbins)
ybins = np.linspace(-70, 70, nbins)

fig0, ax0 = plt.subplots(2, 1, figsize=(12,12))

counts0, *_ = ax0[0].hist2d(data_top[:, 0], data_top[:, 1], bins=[xbins, ybins], cmap='hot', vmin=0)

# fig1, ax1 = plt.subplots(figsize=(12,6))

counts1, *_ = ax0[1].hist2d(data_bottom[:, 0], data_bottom[:, 1], bins=[xbins, ybins], cmap='hot', vmin=0)

counts1_blur = filters.gaussian(counts1, sigma=20)

fig1, ax1 = plt.subplots()

ax1.imshow(counts1_blur, cmap='hot', vmin=0)


ax0[0].set_aspect('equal')
ax0[1].set_aspect('equal')

# ax1.set_aspect('equal')

ax0[0].set_xlim([-70, 70])
ax0[1].set_ylim([-70, 70])

plt.tight_layout()

# tifffile.imwrite(method + '_top_ring.tif', counts0)
# tifffile.imwrite(method + '_bottom_ring.tif', counts1)

tifffile.imwrite(method + '_bottom_ring.tif', counts1_blur)




####### side view #########

data_bottom = data_bottom[data_bottom[:, 1] > 70]

counts2, *_ = ax0[1].hist2d(data_bottom[:, 0], data_bottom[:, 2], bins=[xbins, ybins], cmap='hot', vmin=0)