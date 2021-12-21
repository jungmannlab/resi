#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:47:08 2021

@author: Luciano A. Masullo
"""

import numpy as np
import h5py


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


    """
    Saving data
    """
    
    hf = h5py.File(path + hdf5_fname, 'w')
    hf.create_dataset('locs', data=locs)
    hf.close()

    ''' 
    YAML Saver
    '''
    
    yaml_oldname = path + hdf5_oldname.replace('.hdf5', '.yaml')
    yaml_newname = path + hdf5_fname.replace('.hdf5', '.yaml')
    
    yaml_file_info = open(yaml_oldname, 'r')
    yaml_file_data = yaml_file_info.read()
    
    yaml_newfile = open(yaml_newname, 'w')
    yaml_newfile.write(yaml_file_data)
    yaml_newfile.close()   
    
    print('New Picasso-compatible .hdf5 file and .yaml file successfully created.')