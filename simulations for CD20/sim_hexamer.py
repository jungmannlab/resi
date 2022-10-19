#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:49:42 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import configparser
from datetime import datetime
import yaml
import h5py
import os.path

plt.close('all')

π = np.pi

# =============================================================================
# functions to save the simulated data as a Picasso file
# =============================================================================

# adapted from Picasso io.py
def save_info(path, info, default_flow_style=False):
    with open(path, "w") as file:
        yaml.dump_all(info, file, default_flow_style=default_flow_style)

# adapted from Picasso io.py
def save_locs(path, locs, info):
    #locs = _lib.ensure_sanity(locs, info)
    with h5py.File(path, "w") as locs_file:
        locs_file.create_dataset("locs", data=locs)
    base, ext = os.path.splitext(path)
    info_path = base + ".yaml"
    save_info(info_path, info)

def save_pos(output_path, filename, width, height, pos, info):
    # Save coordinates to csv or Picasso hdf5 file

    # filename without '.hdf5'
    frames = np.full(len(pos), int(0))
    x = pos[:, 0]
    y = pos[:, 1]
    lpx = np.full(len(pos), 0.001) # Dummy value required for Picasso Render to display points
    lpy = np.full(len(pos), 0.001)       
    
    LOCS_DTYPE = [
         ("frame", "u4"),
         ("x", "f4"),
         ("y", "f4"),
         ("lpx", "f4"),
         ("lpy", "f4"),
         ]
    
    locs = np.rec.array(
         (frames, x, y, lpx, lpy),
         dtype=LOCS_DTYPE,
         )

    save_locs(os.path.join(output_path, filename + '.hdf5'), locs, info)

# =============================================================================
# experimental parameters
# =============================================================================

# independent parameters

D = 2 # dimension of the simulation, d = 2 for 2D case, d = 3 for 3D
mult = 6 # multiplicity of the molecular assembly (e.g. mult = 2 for dimers)

R = 20 # real dimer distance in nm
dθ = 1.3 * π/6

density_hex = 100e-6 # molecules per nm^2 (or nm^3)
density_m = 0e-6 # molecules per nm^2 (or nm^3)

σ_label = 5.5 / np.sqrt(2) # nm
width = 20e3 # width of the simulated area in nm
height = 20e3 # height of the simulated area in nm

hex_color = '#009FB7'
mon_color = '#FE4A49'

# distribution = 'evenly spaced'
distribution = 'uniform'

# labelling correction
labelling = True
p = 0.5

# dependent parameters

N_hex = int(density_hex/6 * width * height) # divided by two because N_d it's the number of centers of dimers
N_m = int(density_m * width * height)

PLOTS = True

# =============================================================================
# simulate molecules positions and calculate distances
# =============================================================================

c_pos_hex = np.zeros((N_hex, D)) # initialize array of central positions for dimers
c_pos_mon = np.zeros((N_m, D)) # initialize array of central positions for monomers

if D == 2:
    
    if distribution == 'uniform':
        c_pos_hex[:, 0], c_pos_hex[:, 1] = [np.random.uniform(0, width, N_hex), 
                                            np.random.uniform(0, height, N_hex)]
        
        c_pos_mon[:, 0], c_pos_mon[:, 1] = [np.random.uniform(0, width, N_m), 
                                            np.random.uniform(0, height, N_m)]
        
    else:
        print('Please enter a valid distribution key')
  
if PLOTS:
    
    fig0, ax0 = plt.subplots() # dimers
    fig0.suptitle('Hexamers + their center + monomers')
    # fig1, ax1 = plt.subplots() # monomers
    # fig1.suptitle('Monomers')
              
    ax0.scatter(c_pos_hex[:, 0], c_pos_hex[:, 1], alpha=0.5, marker='*')
    
    ax0.set_xlabel('x (nm)')
    ax0.set_ylabel('y (nm)')
    ax0.set_title('Real density hex = '+str(int(density_hex*1e6))+'/$μm^2$')
    ax0.set_box_aspect(1)
    
    ax0.scatter(c_pos_mon[:, 0], c_pos_mon[:, 1], alpha=0.5, color=mon_color)
    
    # ax0.set_xlabel('x (nm)')
    # ax0.set_ylabel('y (nm)')
    # ax0.set_title('Real density = '+str(int(density_m*1e6))+'/$μm^2$')
    # ax0.set_box_aspect(1)
    
    
# generate angles and positions for each hexamer, total N_hex hexamers

θ_offset = np.random.uniform(0, 2*π, N_hex) # generate random hexamer rotations
θ_offset = np.repeat(θ_offset, mult)

# θ_offset = 0

θ = np.tile(np.array([0, 0, (2/3)*π, (2/3)*π, (4/3)*π, (4/3)*π]), int(N_hex)) + np.tile(np.array([-dθ/2, dθ/2, -dθ/2, dθ/2, -dθ/2, dθ/2]), int(N_hex))

x = R * np.cos(θ + θ_offset)
y = R * np.sin(θ + θ_offset)

x = np.random.normal(loc=x, scale=σ_label) # distances of molecule 0 to the dimer center
y = np.random.normal(loc=y, scale=σ_label) # distances of molecule 1 to the dimer center

pos_x = x + np.repeat(c_pos_hex[:, 0], mult)
pos_y = y + np.repeat(c_pos_hex[:, 1], mult)

pos_hex = np.array([pos_x, pos_y]).T

# pos_hex = np.array([x, y]) + np.tile(c_pos_hex.T, (2, 6, 1))

if PLOTS:
    
    # this plot should output dimers with its center, and two molecules marked with the same color
    ax0.scatter(pos_x, pos_y, alpha=0.5, 
                color=hex_color)
    
    for i in range(len(c_pos_hex[:, 0])):
        circle1 = plt.Circle((c_pos_hex[i, 0], c_pos_hex[i, 1]), R, fill=False, linestyle='--', color = 'k')
        ax0.add_artist(circle1)
    
    length = 500 # nm, length of the display area for the graph
    
    ax0.set_xlim(width/2, width/2 + length)
    ax0.set_ylim(width/2, width/2 + length)
    
    # ax1.set_xlim(width/2, width/2 + length)
    # ax1.set_ylim(width/2, width/2 + length)

pos_mon = c_pos_mon
pos = np.concatenate((pos_hex, pos_mon)) 

N = mult*N_hex + N_m # total number of molecules before labelling

if labelling:
    
    ids = np.random.choice(np.arange(N), size=int((N)*p), replace=False) # take a random subset of indexes of size N * p
    pos = pos[ids] # take only the labelled positions

if PLOTS:    
    # this plot should output dimers taking into account labelling, molecules with black edge are the ones actually labelled
    ax0.scatter(pos[:, 0], pos[:, 1], facecolors='none', edgecolors='k')
    # pass

### NN calculation ###
    
nbrs = NearestNeighbors(n_neighbors=5).fit(pos) # find nearest neighbours
_distances, _indices = nbrs.kneighbors(pos) # get distances and indices
# distances = _distances[:, 1] # get the first neighbour distances

colors = ['#2880C4', '#97D8C4', '#F4B942', '#363636']
fig_knn, ax_knn = plt.subplots(figsize=(5, 5))

for i in range(4):

    # plot histogram of nn-distance of the simulation
    
    distances = _distances[:, i+1] # get the first neighbour distances
    
    freq, bins = np.histogram(distances, bins=1000, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax_knn.plot(bin_centers, freq, color=colors[i], linewidth=2, 
                label='uniform '+str(i+1)+'st-NN')
    
    plt.tight_layout()
    
ax_knn.set_xlim([0, 100])
ax_knn.set_ylim([0, 0.045])

ax_knn.set_xlabel('K-th nearest-neighbour distance (nm)')
ax_knn.set_ylabel('Frequency')
ax_knn.tick_params(direction='in')
ax_knn.set_box_aspect(1)

# =============================================================================
# save positions of the simulated molecules as a Picasso compatible file
# =============================================================================

path = r'/Users/masullo/Documents/GitHub/RESI/simulations for CD20'
filename = 'simulated_hexamers'

px_size = 130
width  = width/px_size
height = height/px_size

info = {}
info["Generated by"] = "Custom Simulation of hexamers for CD20 evaluation (RESI paper)"
info["Width"] = width # pixel 
info["Height"] = height # pixel
info["Pixelsize"] = px_size # in nm

save_pos(path, filename, width, height, pos/px_size, [info])

# =============================================================================
# Create config file with parameters
# =============================================================================

config = configparser.ConfigParser()
config['params'] = {

'Date and time': str(datetime.now()),
'D': D,
'mult': mult,
'R (nm)': R,
'dθ (rad)': dθ,
'density_hex (nm^-2)': density_hex,
'density_m (nm^-2)': density_m,
'σ_label (nm)': σ_label,
'width (nm)': width,
'height (nm)': height,
'distribution': distribution,
'labeling efficiency': p
}

with open(filename + '_params.txt', 'w') as configfile:
    config.write(configfile)


 