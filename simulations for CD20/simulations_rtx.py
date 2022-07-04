#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:58:57 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multinomial, truncnorm
from scipy.special import comb

plt.close('all')

def prob_func(k, n, p):
    
    return comb(n-1, k-1, exact=True) * p**k * (1-p)**(n-k)
    
### parameters ###

p = 0.5
d1 = 12
R = 20

n_terms = 17 # number of prob terms to be calculated

# TODO: think about a way to justify this, sum of uncertainty?

σ_1nn = 8
σ_2nn = 10
σ_3nn = 15
σ_4nn = 20

# σ_1nn = 6
# σ_2nn = 6
# σ_3nn = 6
# σ_4nn = 6

d_array = np.array([0, d1, R, d1+R, d1+R, 
                    2*d1+R, 2*d1+2*R, 2*d1+2*R, 
                    3*d1 + 2*R, 3*d1+3*R, 3*d1+3*R, 
                    4*d1 + 3*R, 4*d1+4*R, 4*d1+4*R,
                    5*d1 + 4*R, 5*d1+5*R, 5*d1+5*R])

prob_1nn = np.zeros(n_terms)
prob_2nn = np.zeros(n_terms)
prob_3nn = np.zeros(n_terms)
prob_4nn = np.zeros(n_terms)


sample_size = 20000000

### 1 st NN ###

for i in range(n_terms):
    
    prob_1nn[i] = prob_func(k=1, n=i, p=p)

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

for i in range(n_terms):
    
    prob_2nn[i] = prob_func(k=2, n=i, p=p)

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

### 3 rd NN ###


for i in range(n_terms):
    
    prob_3nn[i] = prob_func(k=3, n=i, p=p)

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

for i in range(n_terms):
    
    prob_4nn[i] = prob_func(k=4, n=i, p=p)
    
### simulate histogram of 4 th NN ###

prob_4nn_norm = prob_4nn/np.sum(prob_4nn)
p_4nn_array = multinomial.rvs(n=sample_size, p=prob_4nn_norm)

samples_4nn = len(p_4nn_array)*[0]

myclip_a = 0
myclip_b = np.inf
my_std = σ_4nn

for i, (pp, d) in enumerate(zip(p_4nn_array, d_array)):
    a, b = (myclip_a - d) / my_std, (myclip_b - d) / my_std
    samples_4nn[i] = truncnorm.rvs(a, b, loc=d, scale=σ_4nn, size=pp)
    
samples_4nn = np.concatenate(samples_4nn, axis=0)

fig3, ax3 = plt.subplots(figsize=(6,3))

freq_4nn, bins_4nn, _ = ax3.hist(samples_4nn, bins=100, density=True)
ax3.axvline(x=2*d1+2*R, zorder=10, color='red')

ax3.set_xlim([0, 100])
ax3.set_ylim([0, 0.055])
ax3.set_xlabel('3rd NN distance (nm)')
ax3.set_ylabel('Norm Frequency')

#######

fig_knn, ax_knn = plt.subplots(figsize=(12,6))

ax_knn.set_xlim([0, 100])
ax_knn.set_ylim([0, 0.055])
ax_knn.set_xlabel('Kth NN distance (nm)')
ax_knn.set_ylabel('Norm Frequency')

colors = ['#4059AD', '#97D8C4', '#F4B942', '#363636']

ax_knn.plot(bins_1nn[:-1], freq_1nn, color=colors[0])
ax_knn.plot(bins_2nn[:-1], freq_2nn, color=colors[1])
ax_knn.plot(bins_3nn[:-1], freq_3nn, color=colors[2])
ax_knn.plot(bins_4nn[:-1], freq_4nn, color=colors[3])

ax_knn.tick_params(direction='in')

plt.tight_layout()

# =============================================================================
# plot experimental data for comparison
# =============================================================================


filename = 'NND data RTX/well2_CHO_A3_GFPAlfaCd20_RTXAlexa647_GFPNb_apicked_varsD9_15.npz_RESI.npz'

data = dict(np.load(filename))

x = data['new_com_x_cluster']*130
y = data['new_com_y_cluster']*130

pos = np.array([x, y]).T

from sklearn.neighbors import NearestNeighbors

### NN calculation ###
    
nbrs = NearestNeighbors(n_neighbors=6).fit(pos) # find nearest neighbours
_distances, _indices = nbrs.kneighbors(pos) # get distances and indices
# distances = _distances[:, 1] # get the first neighbour distances

colors = ['#4059AD', '#97D8C4', '#F4B942', '#363636']
# fig_knn, ax_knn = plt.subplots(figsize=(5, 5))

for i in range(4):

    # plot histogram of nn-distance of the simulation
    
    distances = _distances[:, i+1] # get the first neighbour distances
    
    # freq, bins = np.histogram(distances, bins=100, density=True)
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # ax_knn.plot(bin_centers, freq, color=colors[i], linewidth=2, 
    #             label='uniform '+str(i+1)+'st-NN')
    
    bins = np.arange(0, 1000, 2)
    ax_knn.hist(distances, bins=bins, alpha=0.7, color=colors[i], edgecolor='black', linewidth=0.1, density=True)

    # ax_knn.set_xlim([0, 200])
    # ax_knn.set_ylim([0, 0.022])
    
    ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
    ax_knn.set_ylabel('Frequency')
    ax_knn.tick_params(direction='in')
    ax_knn.set_box_aspect(1)
    
    plt.tight_layout()
    
# ax_knn.set_xlim([0, 100])
# ax_knn.set_ylim([0, 0.022])

ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
ax_knn.set_ylabel('Frequency')
ax_knn.tick_params(direction='in')
ax_knn.set_box_aspect(1)

