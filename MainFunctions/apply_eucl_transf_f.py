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
import re

import tools

def apply_eucl_transf_f(folder_path, ch1_files, ch3_files):
    plt.close('all')

    plt.close('all')

    plt.close('all')

    px_size = 130 # nm

    eucl_transf = pd.read_csv(os.path.join(folder_path,"eucl_transf/eucl_transf_data.csv"), delimiter='\t')
    picks_array = eucl_transf['pick'].to_numpy()
    #rot_array = eucl_transf['rotation'].to_numpy()
    #tr_array = np.array([eucl_transf['translation x'].to_numpy(), 
    #                      eucl_transf['translation y'].to_numpy()]).T


    for i, (file1, file3) in enumerate(zip(ch1_files, ch3_files)):
        match1 = re.search("ori(\d+)", file1)
        match3 = re.search("ori(\d+)", file3)
        if match1 and match3:
            if match1.group(1) == match3.group(1):
                pick = int(match1.group(1))
        
            else:
                raise Exception('Origamis from channel 1 and channel 3 are not assigned correctly to each other.')
        else:
            raise Exception('Origamis from channel 1 and channel 3 cannot be assigned to each other. Check if "ori" followed by a number is included in the filenames.')
        #print(pick, "vs", picks_array[i])

        transfo_i = eucl_transf.loc[eucl_transf['pick'] == pick]
        angle = transfo_i['rotation'].values[0]
        translation = np.array([transfo_i['translation x'].values[0], transfo_i['translation y'].values[0]])
        transform = tf.EuclideanTransform(rotation=angle, translation=translation)
        
        
        ch1_fulltable = pd.read_hdf(file1, key = 'locs')    
        ch1_fulltable['x'] = ch1_fulltable['x']*px_size # convert to nm
        ch1_fulltable['y'] = ch1_fulltable['y']*px_size
        
        ch1_locs = np.array([ch1_fulltable['x'].to_numpy(), 
                             ch1_fulltable['y'].to_numpy()]).T
        
        ch3_fulltable = pd.read_hdf(file3, key = 'locs')
        
        #print(ch3_fulltable)

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
        
        #fname1 = 'R1_apicked_ori' + str(i) + '_aligned' + '.hdf5'
        #fname3 = 'R3_apicked_ori' + str(i) + '_aligned' + '.hdf5'
        
        filename3_old = os.path.split(file3)[1]
        filename3_aligned = filename3_old[:-5] + "_aligned.hdf5"
        tools.picasso_hdf5(df=ch3_fulltable, 
                            hdf5_fname=filename3_aligned, 
                            hdf5_oldname=filename3_old, 
                            path=folder_path + "/")

        
        try:
            os.mkdir(folder_path + '/transf_overview')
        
        except OSError:
            print ("transf_overview folder already exists")


        match = re.search("ori(\d+)", filename3_aligned)
        plt.savefig(folder_path + '/transf_overview/' + 'origami' + str(match.group(1)) + '.pdf', format='pdf')
        
        #TODO: save in Picasso compatible dataframe