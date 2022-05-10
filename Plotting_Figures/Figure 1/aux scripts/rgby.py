# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 21:27:27 2022

@author: aszalai
"""

import numpy as np
import matplotlib.pyplot as plt

# define normalized 2D gaussian
def gaus2d(x, y, mx, my, sx, sy):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

green_colormap = np.zeros((256,3))
green_colormap[:,1] = np.reshape(np.linspace(0,1,num=256), (256,))

red_colormap = np.zeros((256,3))
red_colormap[:,0] = np.reshape(np.linspace(0,1,num=256), (256,))

blue_colormap = np.zeros((256,3))
blue_colormap[:,2] = np.reshape(np.linspace(0,1,num=256), (256,))

color = 'yellow' # red, green, blue, or yellow

x = np.arange(0,100,1)
y = np.arange(0,100,1)
x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
z = gaus2d(x, y, 50, 50, 10, 10)

M = np.zeros((len(x),len(y)))
M_color = np.zeros((len(x),len(y),3))

M[x,y] = z # M = image

for i in range(len(x)):
    for j in range(len(y)):
        if color == 'red':
            M_color[i,j,0] = M[i,j]/np.max(M)
        elif color == 'green':
            M_color[i,j,1] = M[i,j]/np.max(M)
        elif color == 'blue':
            M_color[i,j,2] = M[i,j]/np.max(M)
        elif color == 'yellow':
            M_color[i,j,0] = M[i,j]/np.max(M)
            M_color[i,j,1] = M[i,j]/np.max(M)

plt.figure()
plt.imshow(M_color)
