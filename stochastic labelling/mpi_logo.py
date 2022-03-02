#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 12:54:28 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# x = np.random.normal(scale=100, size = 500000)
# y = np.random.normal(scale=100, size = 500000)
  
dx = 5
M = [[0, 0], [0, dx], [0, 2*dx], [0, 3*dx], [0, 4*dx],
     [dx, 3*dx + dx/2], [2*dx, 3*dx], [3*dx, 3*dx + dx/2],
     [4*dx, 4*dx], [4*dx, 3*dx], [4*dx, 2*dx], [4*dx, 1*dx], [4*dx, 0]]

P = [[0, 0], [0, dx], [0, 2*dx], [0, 3*dx], [0, 4*dx],
     [dx, 3*dx + dx/2], [2*dx, 4*dx], [3*dx, 3*dx + dx/2],
     [dx, dx + dx/2], [2*dx, dx], [3*dx, dx + dx/2], [3*dx, 2*dx + dx/2]]

I = [[0, 0], [0, dx], [0, 2*dx], [0, 3*dx], [0, 4*dx]]

# x = np.arange(-20, 20, 10)
# y = np.arange(-20, 20, 10)

logo = np.concatenate((np.array(M), np.array(P) + np.array([6*dx, 0]),
                      np.array(I) + np.array([11*dx, 0])))

fig0, ax0 = plt.subplots(figsize =(10, 7))
ax0.scatter(logo[:, 0], logo[:, 1], marker='x', color='k')

# x = np.array(M)[:, 0]
# y = np.array(M)[:, 1]

colors = np.array(['#279AF1', '#C49991', '#CFF27E', '#131112',
                   '#F1A66A', '#ED217C'])

# multinomial with 6 equal probs, take the sites, and then random choice to see which sites
sites = logo.shape[0]
nrounds = 6

label = np.arange(nrounds)

fig1, ax1 = plt.subplots(2, 3, figsize=(20, 16)) 

for _ , ax in enumerate(ax1.reshape(-1)):
    for i, site in enumerate(logo):
        
        color = np.random.choice(colors)
        
        xloc = np.random.normal(loc=site[0], scale=2, size=50)
        yloc = np.random.normal(loc=site[1], scale=2, size=50)
        
        ax.scatter(xloc, yloc, s=50, marker="o", color=color, alpha=0.5)
        ax.scatter(logo[:, 0], logo[:, 1], marker='x', color='k')
        
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')



# c = 0
# for i in range(13):
        
#         p = np.random.uniform()
                
#         if  p < .16:
#             color = '#CFF27E'
#             c += 1
#         else:
#             color = '#DFDFDF'
            
#         # color = np.random.choice(colors)
#         # marker = np.random.choice(markers)
#         # print(color)

#         xloc = np.random.normal(loc=x[i], scale=4, size=50)
#         yloc = np.random.normal(loc=y[i], scale=4, size=50)
        
#         ax.scatter(xloc, yloc, s=50, marker="o", color=color)
#         ax.scatter(x[i], y[i], marker='x', color='#131112')
            
# print(c)
