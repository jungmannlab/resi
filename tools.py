#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:47:08 2021

@author: Luciano A. Masullo
"""

import numpy as np
import h5py

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
    
    labels = ['frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy']
    df_picasso = df.reindex(columns=labels, fill_value=1)
    
    LOCS_DTYPE = [
        (labels[0], 'u4'),
        (labels[1], 'f4'),
        (labels[2], 'f4'),
        (labels[3], 'f4'),
        (labels[4], 'f4'),
        (labels[5], 'f4'),
        (labels[6], 'f4'),
        (labels[7], 'f4'),
        (labels[8], 'f4'),
    ]
    
    locs = np.rec.array(
        (df_picasso.frame, df_picasso.x, 
         df_picasso.y,df_picasso.photons, 
         df_picasso.sx, df_picasso.sy, 
         df_picasso.bg, df_picasso.lpx, 
         df_picasso.lpy), dtype=LOCS_DTYPE,
        )
    
    '''
    Saving data
    '''
    
    hf = h5py.File(path + hdf5_fname, 'w')
    hf.create_dataset('locs', data=locs)
    hf.close()
    
    #TODO: fix this
    
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