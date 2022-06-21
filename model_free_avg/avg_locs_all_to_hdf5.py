# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:59:58 2022

@author: reinhardt
"""


from numba import cuda
import h5py
import numpy as np
import sys
import pandas as pd
import h5py
import os

def picasso_hdf5_empty_yaml(df, hdf5_fname, hdf5_oldname, path):
    
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

    yaml_newname = path + hdf5_fname.replace('.hdf5', '.yaml')
    
    
    yaml_newfile = open(yaml_newname, 'w')
    yaml_newfile.write("")
    yaml_newfile.close()   
    

path = r'W:\users\reinhardt\RESI\locmofit-avg\250122\all_imagers_raw\analysis'
file = 'avg_locs_all_12.csv'


method = 'RESI'

df = pd.read_csv(os.path.join(path,file), names=['xnm', 'ynm', 'znm', 'locprecnm', 'locprecznm', 'channel', 'layer'])
  
print(df.keys())

df2 = pd.DataFrame()

df2['frame'] = np.full(len(df),0)
df2['x'] = -df['xnm']/130+250
df2['y'] = -df['ynm']/130+250
df2['z'] = df['znm']
df2['photons'] = np.full(len(df),1)
df2['sx'] = df['locprecnm']/130
df2['sy'] = df['locprecnm']/130
df2['lpx'] = df['locprecnm']/130
df2['lpy'] = df['locprecnm']/130

df2 = df2.astype({'frame': 'u4', 'x': 'f4', 'y': 'f4', 'z': 'f4', 'photons': 'f4', 
                        'sx': 'f4', 'sy': 'f4', 'lpx': 'f4','lpy': 'f4'})
df3 = df2.reindex(columns = ['frame', 'x', 'y', 'z', 'photons',
                            'sx', 'sy', 'lpx', 'lpy'], fill_value=1)
    

filename_old = file
filename_new = '%s.hdf5' %(filename_old[:-4])
picasso_hdf5_empty_yaml(df3, filename_new, filename_old, path + "/" )