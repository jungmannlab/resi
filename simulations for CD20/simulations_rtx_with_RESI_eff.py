#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:58:57 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multinomial, truncnorm

plt.close('all')

### parameters ###

p = 0.5
r = 0.75
d1 = 12
R = 12

σ_1nn = 6
σ_2nn = 12 # TODO: think about a way to justify this, sum of uncertainty?
σ_3nn = 18

d_array = np.array([0, d1, R, d1+R, d1+R, 2*d1+R, 2*d1+2*R, 2*d1+2*R])

prob_1nn = np.zeros(8)
prob_2nn = np.zeros(8)
prob_3nn = np.zeros(8)

sample_size = 10000000

### 1 st NN ###

prob_1nn[0] = 0
prob_1nn[1] = p * r
prob_1nn[2] = (1 - p * r) * p * r
prob_1nn[3] = (1 - p * r)**2 * p
prob_1nn[4] = (1 - p * r)**2 * (1 - p) * p 
prob_1nn[5] = (1 - p * r)**2 * (1 - p)**2 * p
prob_1nn[6] = (1 - p * r)**2 * (1 - p)**3 * p
prob_1nn[7] = (1 - p * r)**2 * (1 - p)**4 * p

print(prob_1nn)
print(np.sum(prob_1nn))

### simulate histogram of 1 st NN ###

prob_1nn_norm = prob_1nn/np.sum(prob_1nn)
p_1nn_array = multinomial.rvs(n=sample_size, p=prob_1nn_norm)

samples_1nn = len(p_1nn_array)*[0] # create list to store the simulated distances

# params for the >0 norm distribution 
# see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
myclip_a = 0
myclip_b = np.inf
my_std = σ_1nn

for i, (pp, d) in enumerate(zip(p_1nn_array, d_array)):
    a, b = (myclip_a - d) / my_std, (myclip_b - d) / my_std
    samples_1nn[i] = truncnorm.rvs(a, b, loc=d, scale=σ_1nn, size=pp)
    
samples_1nn = np.concatenate(samples_1nn, axis=0)

fig0, ax0 = plt.subplots(figsize=(6,3))

freq_1nn, bins_1nn, _  = ax0.hist(samples_1nn, bins=100, density=True)
ax0.axvline(x=d1, zorder=10, color='red')

ax0.set_xlim([0, 100])
ax0.set_ylim([0, 0.055])
ax0.set_xlabel('1st NN distance (nm)')
ax0.set_ylabel('Norm Frequency')

plt.tight_layout()

### 2 nd NN ###

prob_2nn[0] = 0
prob_2nn[1] = 0
prob_2nn[2] = (p * r) * (p * r)
prob_2nn[3] = 2 * p * (1 - p * r) * p * r
prob_2nn[4] = (1 - p * r) * (1 - p * r) * p * p + (1 - p * r) * (1 - p) * p * r * p * r + p * r * (1 - p * r) * (1 - p) * p

A = p * r
B = (1 - p * r)
C = p
D = (1 - p)

prob_2nn[5] = (B*B*D*C*C)+(B*B*C*D*A)+(B*A*C*D*A)+(A*B*D*D*C)
prob_2nn[6] = 1 - prob_2nn[:6].sum()
prob_2nn[7] = 0

print(prob_2nn)
print(np.sum(prob_2nn))

### simulate histogram of 2 nd NN ###

prob_2nn_norm = prob_2nn/np.sum(prob_2nn)
p_2nn_array = multinomial.rvs(n=sample_size, p=prob_2nn_norm)

samples_2nn = len(p_2nn_array)*[0]

myclip_a = 0
myclip_b = np.inf
my_std = σ_2nn

for i, (pp, d) in enumerate(zip(p_2nn_array, d_array)):
    a, b = (myclip_a - d) / my_std, (myclip_b - d) / my_std
    samples_2nn[i] = truncnorm.rvs(a, b, loc=d, scale=σ_2nn, size=pp)
    
samples_2nn = np.concatenate(samples_2nn, axis=0)

fig1, ax1 = plt.subplots(figsize=(6,3))

freq_2nn, bins_2nn, _  = ax1.hist(samples_2nn, bins=100, density=True)
ax1.axvline(x=d1+R, zorder=10, color='red')

ax1.set_xlim([0, 100])
ax1.set_ylim([0, 0.055])
ax1.set_xlabel('2nd NN distance (nm)')
ax1.set_ylabel('Norm Frequency')

plt.tight_layout()


### 3 rd NN ###


prob_3nn[0] = 0
prob_3nn[1] = 0
prob_3nn[2] = 0

prob_3nn[3] = (p * r) * (p * r) * (p * r)

A = p * r
B = (1 - p * r)
C = p
D = (1 - p)

prob_3nn[4] = (B*A*C*A)+(A*B*A*C)+(A*A*B*A)
prob_3nn[5] = (A*A*B*B*C)+(A*B*A*D*A)+(A*B*B*C*C)+(B*A*C*B*A)+(B*A*D*A*C)+(B*B*C*C*A)

from scipy.special import comb

prob_3nn[6] = comb(5,2) * p**2 * (1-p)**4
prob_3nn[7] = comb(6,2) * p**2 * (1-p)**5
# prob_3nn[8] = comb(7,2) * p**2 * (1-p)**4
# prob_3nn[9] = comb(8,2) * p**2 * (1-p)**5

print(prob_3nn)
print(np.sum(prob_3nn))

### simulate histogram of 3 rd NN ###

prob_3nn_norm = prob_3nn/np.sum(prob_3nn)
p_3nn_array = multinomial.rvs(n=sample_size, p=prob_3nn_norm)

samples_3nn = len(p_3nn_array)*[0]

myclip_a = 0
myclip_b = np.inf
my_std = σ_3nn

for i, (pp, d) in enumerate(zip(p_3nn_array, d_array)):
    a, b = (myclip_a - d) / my_std, (myclip_b - d) / my_std
    samples_3nn[i] = truncnorm.rvs(a, b, loc=d, scale=σ_3nn, size=pp)
    
samples_3nn = np.concatenate(samples_3nn, axis=0)

fig2, ax2 = plt.subplots(figsize=(6,3))

freq_3nn, bins_3nn, _ = ax2.hist(samples_3nn, bins=100, density=True)
ax2.axvline(x=2*d1+R, zorder=10, color='red')

ax2.set_xlim([0, 100])
ax2.set_ylim([0, 0.055])
ax2.set_xlabel('3rd NN distance (nm)')
ax2.set_ylabel('Norm Frequency')


### 4 th NN ###

#######

fig, ax = plt.subplots(figsize=(6,3))

ax.set_xlim([0, 100])
ax.set_ylim([0, 0.055])
ax.set_xlabel('Kth NN distance (nm)')
ax.set_ylabel('Norm Frequency')

ax.plot(bins_1nn[:-1], freq_1nn)
ax.plot(bins_2nn[:-1], freq_2nn)
ax.plot(bins_3nn[:-1], freq_3nn)
# ax.plot(freq_1nn, bins_1nn[:-1])
