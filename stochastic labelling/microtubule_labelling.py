#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:22:48 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

period = 8 # in nm
d = 30 # in nm

N = 3
t = np.linspace(0, 30 * N, 50 * N)
x = (d/2) * np.sin(2*np.pi*t/period)
y = (d/2) * np.cos(2*np.pi*t/period)
z = t

fig0 = plt.figure()
ax0 = fig0.add_subplot(projection='3d')
ax0.scatter3D(x,y,z, s=100)
ax0.plot(x,y,z, alpha=0.1)
ax0.set_xlabel('x (nm)')
ax0.set_ylabel('y (nm)')
ax0.set_zlabel('z (nm)')

ax0.set_box_aspect((1, 1, N))

eff = 0.5
gfp = np.random.choice(np.arange(len(x)), size=int(len(x)*eff), replace=False)

label_eff = 0.6
subset_size = int(len(gfp)*label_eff)
gfp_labelled = np.random.choice(gfp, size=subset_size, replace=False)

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.scatter3D(x[gfp_labelled],y[gfp_labelled],z[gfp_labelled], s=100)
ax1.plot(x,y,z, linestyle='--', alpha=0.1)
ax1.set_xlabel('x (nm)')
ax1.set_ylabel('y (nm)')
ax1.set_zlabel('z (nm)')

ax1.set_box_aspect((1, 1, N))