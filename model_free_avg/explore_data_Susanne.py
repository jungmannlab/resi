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
import os

plt.close('all')

locs = {}

path = r'W:\users\reinhardt\RESI\locmofit-avg\Eduard_analysis'
#iteration = 8

method = 'RESI'

"""for exploration of iterations"""

for i in range(1,11):
    
    locs[str(i)] = pd.read_csv(path + '/' + 'avg_locs_'+str(i)+'.csv')
    
    iteration = i
    data = np.array(locs[str(iteration)])
    
    
    
    
    # Image of top and bottom ring seperatetly
    data_top = data[data[:, 2] > 0]
    data_bottom = data[data[:, 2] < 0]
    nbins = 1400
    xbins = np.linspace(-70, 70, nbins)
    ybins = np.linspace(-70, 70, nbins)
    
    fig0, ax0 = plt.subplots(2, 1, figsize=(12,12))
    
    counts0, *_ = ax0[0].hist2d(data_top[:, 0], data_top[:, 1], bins=[xbins, ybins], cmap='hot', vmin=0)
    
    # fig1, ax1 = plt.subplots(figsize=(12,6))
    
    counts1, *_ = ax0[1].hist2d(data_bottom[:, 0], data_bottom[:, 1], bins=[xbins, ybins], cmap='hot', vmin=0)
    
    #print(counts1)
    ax0[0].set_aspect('equal')
    ax0[1].set_aspect('equal')
    
    # ax1.set_aspect('equal')
    
    ax0[0].set_xlim([-70, 70])
    ax0[1].set_ylim([-70, 70])
    
    plt.tight_layout()
    
    
    plt.savefig(os.path.join(path, 'top_bottom_hist' + str(nbins) + '_iteration_' + str(iteration) + '.png'))
    
    
    ####### side view #########
    
    data_bottom = data_bottom[data_bottom[:, 1] > 70]
    
    counts2, *_ = ax0[1].hist2d(data_bottom[:, 0], data_bottom[:, 2], bins=[xbins, ybins], cmap='hot', vmin=0)
    
    
    # Gaussian filter
    sigma = 20
    counts0_blur = filters.gaussian(counts0, sigma=sigma)
    
    fig_top, ax_top = plt.subplots()
    ax_top.imshow(counts0_blur, cmap='hot', vmin=0)
    plt.savefig(os.path.join(path, 'top_ring_hist' + str(nbins) + '_Gaussian_Filter_sigma' + str(sigma) + '_iteration_' + str(iteration) + '.png'))
    
    
    counts1_blur = filters.gaussian(counts1, sigma=sigma)
    
    fig_bot, ax_bot = plt.subplots()
    ax_bot.imshow(counts1_blur, cmap='hot', vmin=0)
    
    # tifffile.imwrite(method + '_top_ring.tif', counts0)
    # tifffile.imwrite(method + '_bottom_ring.tif', counts1)
    
    #tifffile.imwrite(path + '/' +'bottom_ring.tif', counts1_blur)
    
    plt.savefig(os.path.join(path, 'bottom_ring_hist' + str(nbins) + '_Gaussian_Filter_sigma' + str(sigma) + '_iteration_' + str(iteration) + '.png'))
    

    
"""plot specific iteration"""
"""
iteration = 13
    
locs[str(i)] = pd.read_csv(path + '/' + 'avg_locs_'+str(i)+'.csv')

iteration = i
data = np.array(locs[str(iteration)])




fig = plt.figure(figsize=(20,10))
plt.suptitle('Scatter plot: Top and side view', fontsize = 20)

ax = fig.add_subplot(121, title='x-y scatter plot')
ax.scatter(data[:,0],data[:,1], color='g', s = 0.002)
ax.set_aspect('equal')
ax.set(xlabel='x [nm]', ylabel='y [nm]')


ax = fig.add_subplot(122, title='y-z scatter plot')
ax.scatter(data[:,1],data[:,2], color='g', s = 0.002)
ax.set_aspect('equal')
ax.set(xlabel='y [nm]', ylabel='z [nm]')

plt.show()
"""


# Image of top and bottom ring seperatetly
"""
data_top = data[data[:, 2] > 0]
data_bottom = data[data[:, 2] < 0]
nbins = 1400
xbins = np.linspace(-70, 70, nbins)
ybins = np.linspace(-70, 70, nbins)

fig0, ax0 = plt.subplots(2, 1, figsize=(12,12))

counts0, *_ = ax0[0].hist2d(data_top[:, 0], data_top[:, 1], bins=[xbins, ybins], cmap='hot', vmin=0)

# fig1, ax1 = plt.subplots(figsize=(12,6))

counts1, *_ = ax0[1].hist2d(data_bottom[:, 0], data_bottom[:, 1], bins=[xbins, ybins], cmap='hot', vmin=0)

#print(counts1)
ax0[0].set_aspect('equal')
ax0[1].set_aspect('equal')

# ax1.set_aspect('equal')

ax0[0].set_xlim([-70, 70])
ax0[1].set_ylim([-70, 70])

plt.tight_layout()


plt.savefig(os.path.join(path, 'top_bottom_hist' + str(nbins) + '_iteration_' + str(iteration) + '.png'))


####### side view #########

data_bottom = data_bottom[data_bottom[:, 1] > 70]

counts2, *_ = ax0[1].hist2d(data_bottom[:, 0], data_bottom[:, 2], bins=[xbins, ybins], cmap='hot', vmin=0)


# Gaussian filter
sigma = 20
counts0_blur = filters.gaussian(counts0, sigma=sigma)

fig_top, ax_top = plt.subplots()
ax_top.imshow(counts0_blur, cmap='hot', vmin=0)
plt.savefig(os.path.join(path, 'top_ring_hist' + str(nbins) + '_Gaussian_Filter_sigma' + str(sigma) + '_iteration_' + str(iteration) + '.png'))


counts1_blur = filters.gaussian(counts1, sigma=sigma)

fig_bot, ax_bot = plt.subplots()
ax_bot.imshow(counts1_blur, cmap='hot', vmin=0)

# tifffile.imwrite(method + '_top_ring.tif', counts0)
# tifffile.imwrite(method + '_bottom_ring.tif', counts1)

#tifffile.imwrite(path + '/' +'bottom_ring.tif', counts1_blur)

plt.savefig(os.path.join(path, 'bottom_ring_hist' + str(nbins) + '_Gaussian_Filter_sigma' + str(sigma) + '_iteration_' + str(iteration) + '.png'))



"""
