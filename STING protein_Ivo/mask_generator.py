#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:32:16 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage import filters

plt.close('all')

path = r'well6_resting_RESI/'
filename = r'All_RESI_centers_noZ.hdf5'

filepath = os.path.join(path, filename)
df = pd.read_hdf(filepath, key = 'locs')

x = df.x*130
y = df.y*130

pos_exp = np.array([x, y]).T

# =============================================================================
# Input parameteres for the FOV of the data
# =============================================================================

x0 = 20000
y0 = 33000
length = 12000

fig0, ax0 = plt.subplots()
ax0.set(facecolor='black')
# fig0.suptitle('Monomers + dimers not distinguishable')

ax0.scatter(pos_exp[:, 0], pos_exp[:, 1], facecolors='orange', edgecolors='none', s=2)

ax0.set_xlabel('x (nm)')
ax0.set_ylabel('y (nm)')
ax0.set_xlim(x0, x0 + length)
ax0.set_ylim(y0, y0 + length)
ax0.set_title('Scatter plot of experimental data')
ax0.set_box_aspect(1)

# ax0.set_xlim([22000, 34000])
# ax0.set_ylim([33000, 45000])

fig1, ax1 = plt.subplots()

x0_hist = 15000
y0_hist = 5000
length_hist = 60015
binsize = 20 # in nm
bins_x = np.arange(x0_hist, x0_hist + length_hist, binsize)
bins_y = np.arange(y0_hist, y0_hist + length_hist, binsize)
counts, xedges, yedges, *_ = ax1.hist2d(x, y, bins=[bins_x, bins_y], cmap='hot')

# ax1.set_xlim(x0, x0 + length)
# ax1.set_ylim(y0, y0 + length)

# ax1.set_xlim([22000, 34000])
# ax1.set_ylim([33000, 45000])
ax1.set_box_aspect(1)

image_blurred = filters.gaussian(counts, sigma=4)
image_blurred = np.rot90(image_blurred) #TODO: check why this operation is needed

fig2, ax2 = plt.subplots()

ax2.set_box_aspect(1)

ax2.set_xlabel('x (nm)')
ax2.set_ylabel('y (nm)')
# ax2.set_xlim([22000, 34000])
# ax2.set_ylim([33000, 45000])
ax2.set_xlim(x0, x0 + length)
ax2.set_ylim(y0, y0 + length)
ax2.set_title('Blurred experimental data')

ax2.imshow(image_blurred, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')

thresh = filters.threshold_otsu(image_blurred)
# thresh = image_blurred[image_blurred > 0].mean() + 0.8 * image_blurred[image_blurred > 0].std()

mask = image_blurred >= thresh

fig3, ax3 = plt.subplots()

ax3.set_box_aspect(1)

ax3.set_title('Binary mask')

ax3.imshow(mask, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')

ax3.set_xlabel('x (nm)')
ax3.set_ylabel('y (nm)')
# ax3.set_xlim([22000, 34000])
# ax3.set_ylim([33000, 45000])
ax3.set_xlim(x0, x0 + length)
ax3.set_ylim(y0, y0 + length)

np.save('mask_' + filename, mask)