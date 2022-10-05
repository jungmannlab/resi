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
import scipy.ndimage as ndi
import configparser
from datetime import datetime

plt.close('all')

# Ivo STING

# path = r'well6_resting_RESI/'
# filename = r'All_RESI_centers_noZ_picked.hdf5'

# mask_name = r'All_RESI_centers_noZ_picked' + '_MASK'

# Susi FA

path = r'forLuciano/'
filename = r'K2_picked_mask_in_resi_7_15.hdf5'

mask_name = r'K2_picked_mask_in_resi_7_15' + '_MASK'

filepath = os.path.join(path, filename)
df = pd.read_hdf(filepath, key = 'locs')

px_size = 130 # in nm
x = df.x*px_size
y = df.y*px_size

pos_exp = np.array([x, y]).T

### parameters for the masking

# Note: inspect the mask result to see if it is in good agreement with intuition
# Parameters binsize and σ can be varied but their product should be kept the same
# to obtain similar results (rule of thumb). Binsize should always be greater
# than mask_resolution and they should be multiples.

margin = 50 # in nm, size of the added empty margin to the FOV
binsize = 10 # in nm, size of the 2D histogram of the first step
σ = 6 # in nm, parameter of the gaussian blur in "binsize" units
mask_resolution = 5 # in nm, controls the digital resolution of the mask

# =============================================================================
# Input parameteres for the FOV of the data
# =============================================================================

# create proper roi

x0 = x.min() - margin
y0 = y.min() - margin
length = np.max((x.max() - x.min(),
                 y.max() - y.min())) + 2*margin

fig0, ax0 = plt.subplots()
ax0.set(facecolor='black')

ax0.scatter(pos_exp[:, 0], pos_exp[:, 1], facecolors='orange', edgecolors='none', s=4)

ax0.set_xlabel('x (nm)')
ax0.set_ylabel('y (nm)')
ax0.set_xlim(x0, x0 + length)
ax0.set_ylim(y0, y0 + length)
ax0.set_title('Scatter plot of experimental data')
ax0.set_box_aspect(1)

fig1, ax1 = plt.subplots()

x0_hist = x0
y0_hist = y0
length_hist = length

bins_x = np.arange(x0_hist, x0_hist + length_hist, binsize)
bins_y = np.arange(y0_hist, y0_hist + length_hist, binsize)
counts, xedges, yedges, *_ = ax1.hist2d(x, y, bins=[bins_x, bins_y], cmap='hot')

ax1.set_box_aspect(1)

image_blurred = filters.gaussian(counts, sigma=σ)
image_blurred = np.rot90(image_blurred) #TODO: check why this operation is needed

fig2, ax2 = plt.subplots()

ax2.set_box_aspect(1)

ax2.set_xlabel('x (nm)')
ax2.set_ylabel('y (nm)')
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
ax3.set_xlim(x0, x0 + length)
ax3.set_ylim(y0, y0 + length)

length_rounded = np.around(length)
factor = int(binsize/mask_resolution)
mask_zoomed = ndi.zoom(np.array(mask, dtype=float), factor)

fig4, ax4 = plt.subplots()

ax4.set_box_aspect(1)
ax4.set_title('Binary mask - upsampled')
ax4.imshow(mask_zoomed, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')

ax4.set_xlabel('x (nm)')
ax4.set_ylabel('y (nm)')
ax4.set_xlim(x0, x0 + length)
ax4.set_ylim(y0, y0 + length)

mask_final = mask_zoomed > 0.5

# upsample of xedges and yedges
xedges = np.arange(xedges[0], xedges[-1], step=mask_resolution)
yedges = np.arange(yedges[0], yedges[-1], step=mask_resolution)

fig5, ax5 = plt.subplots()

ax5.set_box_aspect(1)
ax5.set_title('Binary mask - final')
ax5.imshow(mask_final, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')

ax5.set_xlabel('x (nm)')
ax5.set_ylabel('y (nm)')
ax5.set_xlim(x0, x0 + length)
ax5.set_ylim(y0, y0 + length)

np.save(mask_name + '.npy', mask_final)
np.save(mask_name + '_xedges' + '.npy', xedges)
np.save(mask_name + '_yedges' + '.npy', yedges)

frac_of_area_with_molec = mask_final.sum()/mask_final.shape[0]**2
mask_area = mask_final.sum() * mask_resolution**2 

observed_density = 1e6 * pos_exp.shape[0]/mask_area

print('Frac of area with molecules =', np.around(frac_of_area_with_molec, 3))
print('Observed density = ', observed_density, 'molecules per μm^2')

# create config file with parameters

config = configparser.ConfigParser()

config['params'] = {

'Date and time': str(datetime.now()),
'px_size (nm)': px_size,
'margin (nm)': margin,
'bin size (nm)': binsize,
'gaussian blur sigma (units of bin size)': σ,
'mask digital resolution (nm)': mask_resolution,
'Observed density (μm^-2)': observed_density,
'mask file name': mask_name + '.npy'}

with open(mask_name + '_params.txt', 'w') as configfile:
    config.write(configfile)
