#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:51:17 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

plt.close('all')

def circle(grid, amplitude, x0, y0, r):
    
    n = np.shape(grid[0])[0]
    x, y = grid

    mask = (x-x0)**2 + (y-y0)**2 <= r**2

    array = np.zeros((n, n))
    array[mask] = amplitude
    
    return array

### simulate circular masks

size_c = 500 # in px
nsubimages_c = 10
    
x = np.arange(0, size_c)
y = np.arange(0, size_c) 
    
[Mx, My] = np.meshgrid(x, y)

r = 100 # in px
subimage_c = circle((Mx, My), 1, int(size_c/2), int(size_c/2), r)
image_c = np.tile(subimage_c, (nsubimages_c, nsubimages_c))

fig0, ax0 = plt.subplots()
ax0.imshow(image_c, cmap='gray')

np.save('circular_masks.npy', image_c)

### simulate linear masks
 
size_l = 500 # in px
subimage_l = np.zeros((size_l, size_l))
nsubimages_l = 10

i_start, i_end = [225, 275]
j_start, j_end = [75, 425]

subimage_l[i_start:i_end, j_start:j_end] = 1

image_l = np.tile(subimage_l, (nsubimages_l, nsubimages_l))

fig1, ax1 = plt.subplots()
ax1.imshow(image_l, cmap='gray')

np.save('linear_masks.npy', image_l)
