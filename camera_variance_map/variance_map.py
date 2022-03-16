#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:12:21 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import datetime

plt.close('all')

print(datetime.datetime.now())

calculate = False
plot = True

if calculate:

    # open data
    path = r'/Volumes/pool-miblab4/users/reinhardt/z_raw/VarianceMaps/Skylab_150ms_1/'
    filename = path + 'Skylab_150ms_1_MMStack_Pos0.ome.tif'
    stack = np.array(imageio.mimread(filename, memtest=False))
    
    # check properties of the array
    print(type(stack))
    print(stack.shape)
    
    # calculate mean, var and std
    stack_mean = np.mean(stack, axis=0)
    stack_var = np.var(stack, axis=0)
    stack_std = np.std(stack, axis=0)
    
    # save mean, var and std maps
    np.save('mean.npy', stack_mean) #TODO: check folder
    np.save('std.npy', stack_std)
    np.save('var.npy', stack_var)
    
elif plot:
    
    stack_mean = np.load('mean.npy')
    stack_std = np.load('std.npy')
    stack_var = np.load('var.npy')
    
    #plot histograms of mean, std and var
    fig0, ax0 = plt.subplots()
    fig0.suptitle('Mean values histogram')
    ax0.hist(stack_mean.flatten(), bins=50)

    fig1, ax1 = plt.subplots()
    fig1.suptitle('Std values histogram')
    maxval = 15 # maxval fine tuned to exclude outliers
    minval = 0
    bins = np.linspace(minval, maxval, 100) 
    ax1.hist(stack_std.flatten(), bins=bins)

    fig2, ax2 = plt.subplots()
    fig2.suptitle('Var values histogram')
    maxval = 200 # maxval fine tuned to exclude outliers
    minval = 0
    bins = np.linspace(minval, maxval, 100)
    ax2.hist(stack_var.flatten(), bins=bins)

    #plot 2D maps of mean, std and var
    fig3, ax3 = plt.subplots()
    fig3.suptitle('Mean map')
    ax3.imshow(stack_mean, interpolation='None')

    fig4, ax4 = plt.subplots()
    fig4.suptitle('Std map')
    ax4.imshow(stack_std, interpolation='None', vmin=0, vmax=20)

    fig5, ax5 = plt.subplots()
    fig5.suptitle('Var map')
    ax5.imshow(stack_var, interpolation='None', vmin=0, vmax=400)

print(datetime.datetime.now())








