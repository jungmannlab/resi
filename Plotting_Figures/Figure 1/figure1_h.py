#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:46:52 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

σ_dnapaint = 3.0

kk = np.linspace(1, 100, 1000)

fig, ax = plt.subplots()

ax.plot(kk, σ_dnapaint/np.sqrt(kk))
