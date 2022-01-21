#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:47:08 2021

@author: Luciano A. Masullo
"""

import numpy as np
import pandas as pd
import h5py
import os


def angle(p1x, p1y, p2x, p2y):
    angle = np.zeros(len(p1x))
    for i in range(len(p1x)):
        p1 = np.array([p1x[i], p1y[i]])
        p2 = np.array([p2x[i], p2y[i]])
        """
        if p1y[i] > p2y[i]:
            p3x = p1x[i] + 1
            p3y = p1y[i]
            p3 = np.array([p3x, p3y])
            v0 = p3 - p1
            v1 = p2 - p1
        else:
            p3x = p2x[i] + 1
            p3y = p2y[i]
            p3 = np.array([p3x, p3y])
            v0 = p3 - p2
            v1 = p2 - p2
        """
        p3x = p1x[i] + 1
        p3y = p1y[i]
        p3 = np.array([p3x, p3y])
        v0 = p3 - p1
        v1 = p2 - p1
        angle[i] = np.degrees(np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1)))
    return angle
            
        
        
        
        
def picasso_hdf5(df, hdf5_fname, hdf5_oldname, path):
    
    """
    This function recieves a pandas data frame coming from a Picasso .hdf5
    file but that has been modified somehow (e.g. coordinates rotated, 
    groups edited, etc) and formats it to be compatible with Picasso. 
    It also creates the necessary .yaml file.
    
    It is meant to be used in a folder that contains the original Picasso
    .hdf5 file and the .yaml file.
    
    - df: pandas data frame object
    - hdf5_fname: the desired filename for the Picasso-compatible .hdf5 file
    - hdf5_oldname: name of the original Picasso file that was modified
    - path: the absolute path containing the path to the file's folder
    
    Note: include ".hdf5" in the file names
    Warning: the .yaml file is basically a copy of the old one. If important
    information should be added to the new .yaml file this function should be
    modified
    
    """

    labels = list(df.keys())
    df_picasso = df.reindex(columns=labels, fill_value=1)
    locs = df_picasso.to_records(index = False)

    # Saving data
    
    hf = h5py.File(path + hdf5_fname, 'w')
    hf.create_dataset('locs', data=locs)
    hf.close()

    # YAML saver

    yaml_oldname = path + hdf5_oldname.replace('.hdf5', '.yaml')
    yaml_newname = path + hdf5_fname.replace('.hdf5', '.yaml')
    
    yaml_file_info = open(yaml_oldname, 'r')
    yaml_file_data = yaml_file_info.read()
    
    yaml_newfile = open(yaml_newname, 'w')
    yaml_newfile.write(yaml_file_data)
    yaml_newfile.close()   
    
    print('New Picasso-compatible .hdf5 file and .yaml file successfully created.')
    
    
def get_resi_locs(files, K):
    
    """
    input:
        
        files: must be a list of strings containing the file names by channel
        K: number of localizations to be averaged to obtain each resi 
        localization
    
    output: 
        
        data: a data frame with the localizations with one extra property, 
              the 'subset'
        resi_locs: a data frame with the resi localizations obtained by 
                   averaging in the subsets
    """
    
    data = {}
    nclusters = {}
    resi_locs = {}
    
    for i, file in enumerate(files):
        data[str(i)] = pd.read_hdf(file, key='locs') # read each channel (file)
        data[str(i)].rename(columns={'group': 'cluster_id'}, inplace=True)
        
    for key in data.keys():
        
        nclusters[key] = data[key]['cluster_id'].max()+1 # TODO: 0-index the clusters so they're easier to iterate 
        
        # initalize a list with the 'subset' property
        subsetslist = [-1]*data[key].shape[0] # -1 is the label for localizations not assigned to any subset
        data[key]['subset'] = subsetslist

        for i in range(1, nclusters[key]): # TODO: check 0-index vs 1-index in clusters
            
            cluster = data[key].loc[data[key]['cluster_id'] == i] # get the table of cluster i   
            indexes = cluster.index # get the (general) indexes of the localizations in this cluster
            nlocs = cluster.shape[0] # get the number of localizations in cluster i
            nsubsets = int(nlocs/K) # get number of subsets, given K

            for j in range(nsubsets):
                
                # random choice of size K
                subsets_id = np.random.choice(indexes, size=K, replace=False) 
                indexes = [i for i in indexes if i not in subsets_id] # remove already chosen indexes
                data[key].loc[subsets_id, 'subset'] = j # assign a subset label   
                
        grouped_locs = data[key].groupby(['cluster_id', 'subset']) # group localizations by cluster_id and subset
                                                                
        resi_locs[key] = grouped_locs.mean().reset_index() # calculate mean by cluster_id and subset and obtain resi data
    
    return resi_locs, data


def simulate_data(fname, sites, locs_per_site, σ_dnapaint, plot=False):
    
    """
    input:
            sites: array with the coordinates of the docking sites (in nm)
            locs_per_site: number of localizations per site
            σ_dnapaint: DNA-PAINT precision in nm
    
    output: 
            it writes a file with the simulated data

    """
    
    cov = [[σ_dnapaint**2, 0], [0, σ_dnapaint**2]] # create covariance matrix
    locs = locs_per_site # number of localizations per docking site
    
    # generate simulated origami data
    
    data = np.zeros((sites.shape[0], locs, 2))
    xlist = []
    ylist = []
    clusterlist = []
    
    for i, site in enumerate(sites):
    
        data[i, :, :] = np.random.multivariate_normal(site, cov, locs) 
        
        xlist += list(data[i, :, 0])
        ylist += list(data[i, :, 1])
        clusterlist += list(np.array((np.ones(data.shape[1])*i + 1), 
                                     dtype=int))
    
    d = {'x': xlist, 'y': ylist, 'cluster_id': clusterlist}
    df = pd.DataFrame(d)
    df.to_hdf(fname, key='locs', mode='w')
    
    if plot: # plot whole origami (histogram)
    
        import matplotlib.pyplot as plt
    
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel('x (nm)')
        ax1.set_xlim([-50, 50])
        ax1.set_ylim([-50, 50])
        ax1.set_ylabel('y (nm)')
        
        bins = np.arange(-50, 50, 0.2)
        hist, xbins, ybins = np.histogram2d(data[:, :, 0].flatten(), 
                                            data[:, :, 1].flatten(), 
                                            bins=bins)
        extent = [-50, 50, -50, 50]
        
        ax1.imshow(hist.T, interpolation='none', origin='lower', 
                   cmap='hot', 
                   extent=extent)
        ax1.set_aspect('equal')
    
        
    return "Simulated data successfully generated"