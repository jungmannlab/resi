#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:50:04 2021

@author: Luciano A. Masullo

Apply to each origami the previously found eucl transform that optimizes
the distance between the two channels.

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
import pandas as pd
import h5py
import os

import tools

plt.close('all')

plt.close('all')

folder_path = r'/Volumes/pool-miblab4/users/reinhardt/z_raw/Resi/211129_double-ext-20nm-2del-6sites-R4-single-site-ca/aligned/Rotation-test/indiv_origami_locs/'
plt.close('all')

px_size = 130 # nm

eucl_transf = pd.read_csv(folder_path + 'eucl_transf_data.csv', delimiter='\t')
rot_array = eucl_transf['rotation'].to_numpy()
tr_array = np.array([eucl_transf['translation x'].to_numpy(), 
                      eucl_transf['translation y'].to_numpy()]).T

# I will avoid calling them "R1" or "R3" since the data also contains R4 imagers

ch1_files = []
ch3_files = []

# get number of files assuming there are 4 per origami (R1, R2, yaml and hdf5)
n_files = int(len(os.listdir(folder_path))/4) #TODO: improve this to read how many target files are there in the folder

# get all hdf5 file names in a list
for i in range(n_files):

    ch1_files.append('R1_apicked_ori' + str(i) + '.hdf5')
    ch3_files.append('R3_apicked_ori' + str(i) + '.hdf5')
    
for i, (file1, file3) in enumerate(zip(ch1_files, ch3_files)):
    
    print(i)
    
    # load the localizations for each channel

    ch1_fulltable = pd.read_hdf(folder_path + file1, key = 'locs')    
    ch1_fulltable['x'] = ch1_fulltable['x']*px_size # convert to nm
    ch1_fulltable['y'] = ch1_fulltable['y']*px_size
    
    ch1_locs = np.array([ch1_fulltable['x'].to_numpy(), 
                         ch1_fulltable['y'].to_numpy()]).T
    
    ch3_fulltable = pd.read_hdf(folder_path + file3, key = 'locs')
    
    print(ch3_fulltable)

    ch3_fulltable['x'] = ch3_fulltable['x']*px_size # convert to nm
    ch3_fulltable['y'] = ch3_fulltable['y']*px_size
    
    ch3_locs = np.array([ch3_fulltable['x'].to_numpy(), 
                         ch3_fulltable['y'].to_numpy()]).T
    
    # plot the data before eucl transf
    fig, ax = plt.subplots(2, figsize=(12,12))
    
    ax[0].plot(ch1_locs[:, 0], ch1_locs[:, 1], 'o', alpha=0.3, label='Channel 1')
    ax[0].plot(ch3_locs[:, 0], ch3_locs[:, 1], 'o', alpha=0.3, label='Channel 2')
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('x (nm)')
    ax[0].set_ylabel('y (nm)')
    ax[0].legend()
    
    angle = rot_array[i]
    translation = tr_array[i, :]
    transform = tf.EuclideanTransform(rotation=angle, translation=translation)
    
    ch3_locs_tr = transform(ch3_locs)
    
    ch3_fulltable['x'] = ch3_locs_tr[:, 0]
    ch3_fulltable['y'] = ch3_locs_tr[:, 1]
    
    # plot the data after the transformation
    ax[1].plot(ch1_locs[:, 0], ch1_locs[:, 1], 'o', alpha=0.3, label='Channel 1')
    ax[1].plot(ch3_locs_tr[:, 0], ch3_locs_tr[:, 1], 'o', alpha=0.3, label='Channel 2 - transformed')
    ax[1].set_aspect('equal')
    ax[1].set_xlabel('x (nm)')
    ax[1].set_ylabel('y (nm)')
    ax[1].legend()
    
    plt.tight_layout()
    
    # save into a new Picasso-formated hdf5
    
    ch1_fulltable.x = ch1_fulltable['x']/px_size # convert to px
    ch1_fulltable.y = ch1_fulltable['y']/px_size   
    
    ch3_fulltable.x = ch3_fulltable['x']/px_size # convert to px
    ch3_fulltable.y = ch3_fulltable['y']/px_size   
    
    # fname1 = 'R1_apicked_ori' + str(i) + '_aligned' + '.hdf5'
    fname3 = 'R3_apicked_ori' + str(i) + '_aligned' + '.hdf5'
    
    tools.picasso_hdf5(df=ch3_fulltable, 
                        hdf5_fname=fname3, 
                        hdf5_oldname=file3, 
                        path=folder_path)

    
    # ch3_fulltable_picasso = ch3_fulltable.reindex(columns = ['frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy'], fill_value=1)
    
    # LOCS_DTYPE = [
    #     ('frame', 'u4'),
    #     ('x', 'f4'),
    #     ('y', 'f4'),
    #     ('photons', 'f4'),
    #     ('sx', 'f4'),
    #     ('sy', 'f4'),
    #     ('bg', 'f4'),
    #     ('lpx', 'f4'),
    #     ('lpy', 'f4'),
    # ]
    # locs = np.rec.array(
    #     (ch3_fulltable_picasso.frame, ch3_fulltable_picasso.x, 
    #       ch3_fulltable_picasso.y, ch3_fulltable_picasso.photons, 
    #       ch3_fulltable_picasso.sx, ch3_fulltable_picasso.sy, 
    #       ch3_fulltable_picasso.bg, ch3_fulltable_picasso.lpx, 
    #       ch3_fulltable_picasso.lpy), dtype=LOCS_DTYPE,
    #     )
    
    # '''
    # Saving data
    # '''
    
    # hf = h5py.File(folder_path + fname3, 'w')
    # hf.create_dataset('locs', data=locs)
    # hf.close()
    
    # #TODO: fix this
    
    # ''' 
    # YAML Saver
    # '''
    
    # yaml_file_name = folder_path + file3.replace('.hdf5', '.yaml')
    # yaml_file_info = open(yaml_file_name, 'r')
    # yaml_file = yaml_file_info.read()
    
    # yaml_file1 = open(folder_path + fname3.replace('.hdf5', '.yaml'), 'w')
    # yaml_file1.write(yaml_file)
    # yaml_file1.close()   

    
    try:
        os.mkdir(folder_path + '/transf_overview')
    
    except OSError:
        print ("transf_overview folder already exists")

    plt.savefig(folder_path + '/transf_overview/' + 'origami' + str(i), format='pdf')
    
    #TODO: save in Picasso compatible dataframe