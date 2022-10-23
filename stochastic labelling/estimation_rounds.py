#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:56:58 2022

@author: masullo
"""

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt


def prob(n, m):
    """ 
    n: number of orthogonal sequences
    m: number of molecules within DNA-PAINT resolution area
    """
    
    return (factorial(n)/factorial(n -m))/n**m

fig, ax = plt.subplots()

n_array = np.arange(2, 20)

for m in np.arange(0, 6):
    
   ax.plot(n_array, 100 * prob(n_array, m), '-o', label='m = '+str(m))

ax.set_xticks(np.arange(2, 20, 2))
ax.legend()

ax.set_ylabel('Fraction of orth. labeled molecules (%)')
ax.set_xlabel('Number of exchange rounds')

ax.set_xlim(1.8, 19.2)
ax.set_box_aspect(1)