#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Luciano A. Masullo, Susanne Reinhardt

Apply to each origami the previously found eucl transform that minimizes
the distance between alignment siteshe in the two channels.
Input: The previously generated eucl_transf_data.csv
Output: 
    
    
    
    
    
    saved in 'eucl_transf' subfolder
- csv and excel file containing the rotation and translation parameters
- pdf images of each origami showing the overlay of both rounds before and 
  after the alignment.
"""




import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
import pandas as pd
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

        # get the rotation and translation parameters for the current (file1, file3) pair
        transfo_i = eucl_transf.loc[eucl_transf['pick'] == pick]

        
        ch1_fulltable = pd.read_hdf(file1, key = 'locs')    
        ch1_fulltable['x'] = ch1_fulltable['x']*px_size # convert to nm
        ch1_fulltable['y'] = ch1_fulltable['y']*px_size
        try:
            ch1_fulltable['z'] = ch1_fulltable['z'] # already in nm
        except KeyError:
            flag_3D_1 = False
        else:
            flag_3D_1 = True


        ch3_fulltable = pd.read_hdf(file3, key = 'locs')
        ch3_fulltable['x'] = ch3_fulltable['x']*px_size # convert to nm
        ch3_fulltable['y'] = ch3_fulltable['y']*px_size
        try:
            ch3_fulltable['z'] = ch3_fulltable['z'] # already in nm
        except KeyError:
            flag_3D_2 = False
        else:
            flag_3D_2 = True

        if flag_3D_1 == flag_3D_2:
            flag_3D = flag_3D_1
        else:
            raise Exception('The data in both channels has to be either 2D or 3D data.')
        

        if not flag_3D: # 2D
            ch1_locs = np.array([ch1_fulltable['x'].to_numpy(), 
                                 ch1_fulltable['y'].to_numpy()]).T
            ch3_locs = np.array([ch3_fulltable['x'].to_numpy(), 
                                 ch3_fulltable['y'].to_numpy()]).T

            angle = transfo_i['rotation'].values[0]
            translation = np.array([transfo_i['translation x'].values[0], transfo_i['translation y'].values[0]])
            transform = tf.EuclideanTransform(rotation=angle, translation=translation)
            

            
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
            
            ch1_fulltable['x'] = ch1_fulltable['x']/px_size # convert to px
            ch1_fulltable['y'] = ch1_fulltable['y']/px_size   
            
            ch3_fulltable['x'] = ch3_fulltable['x']/px_size # convert to px
            ch3_fulltable['y'] = ch3_fulltable['y']/px_size   

            #fname1 = 'R1_apicked_ori' + str(i) + '_aligned' + '.hdf5'
            #fname3 = 'R3_apicked_ori' + str(i) + '_aligned' + '.hdf5'
            
            filename3_old = os.path.split(file3)[1]
            filename3_aligned = filename3_old[:-5] + "_aligned.hdf5"
            tools.picasso_hdf5(df=ch3_fulltable, 
                                hdf5_fname=filename3_aligned, 
                                hdf5_oldname=filename3_old, 
                                path=folder_path + "/")

        else: # 3D

            # prepare the data in the matrix form (the function rigid_transform_3D(A,B) expects the input arrays as 3xN arrays)

            ch1_locs = np.array([ch1_fulltable['x'].to_numpy(), 
                                 ch1_fulltable['y'].to_numpy(),
                                 ch1_fulltable['z'].to_numpy()])
            ch3_locs = np.array([ch3_fulltable['x'].to_numpy(), 
                                 ch3_fulltable['y'].to_numpy(),
                                 ch3_fulltable['z'].to_numpy()])

            #print(transfo_i)
            #print(transfo_i['rotation matrix'], type(transfo_i['rotation matrix']))
            #print(transfo_i['rotation matrix'].values[0], type(transfo_i['rotation matrix'].values[0]))
            
            rotation_str = transfo_i['rotation matrix'].values[0].replace('\n','').replace('[','').replace(']','').replace('  ',' ')
            rotation = np.fromstring((rotation_str), sep=' ').reshape((3,3))
            #print('rotation', rotation, type(rotation), len(rotation))
            translation_str = transfo_i['translation vector'].values[0].replace('\n','').replace('[','').replace(']','').replace('  ',' ')
            translation = np.fromstring((translation_str), sep=' ').reshape((3,1))
            print('translation', translation, type(translation), type(translation[0]))
            
            # plot the data before eucl transf
            fig = plt.figure(figsize=(24,12))

            ax = fig.add_subplot(241, projection = '3d')
            ax.plot(ch1_locs[0, :], ch1_locs[1, :], ch1_locs[2, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(ch3_locs[0, :], ch3_locs[1, :], ch3_locs[2, :], 'o', alpha=0.3, label='Channel 2')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('y (nm)')
            ax.set_ylabel('z (nm)')
            ax.legend()

            ax = fig.add_subplot(242)
            ax.plot(ch1_locs[0, :], ch1_locs[1, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(ch3_locs[0, :], ch3_locs[1, :], 'o', alpha=0.3, label='Channel 2')
            ax.set_aspect('equal')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('y (nm)')
            ax.legend()

            ax = fig.add_subplot(243)
            ax.plot(ch1_locs[0, :], ch1_locs[2, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(ch3_locs[0, :], ch3_locs[2, :], 'o', alpha=0.3, label='Channel 2')
            ax.set_aspect('equal')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('z (nm)')
            ax.legend()

            ax = fig.add_subplot(244)
            ax.plot(ch1_locs[1, :], ch1_locs[2, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(ch3_locs[1, :], ch3_locs[2, :], 'o', alpha=0.3, label='Channel 2')
            ax.set_aspect('equal')
            ax.set_xlabel('y (nm)')
            ax.set_ylabel('z (nm)')
            ax.legend()


            # transform the data with the opt params 
            ch3_locs_tr = np.linalg.inv(rotation) @ (ch3_locs - translation)
            ch3_fulltable['x'] = ch3_locs_tr[0, :]
            ch3_fulltable['y'] = ch3_locs_tr[1, :]
            ch3_fulltable['z'] = ch3_locs_tr[2, :]
            
            # plot the data after the transformation
            ax = fig.add_subplot(245, projection = '3d')
            ax.plot(ch1_locs[0, :], ch1_locs[1, :], ch1_locs[2, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(ch3_locs_tr[0, :], ch3_locs_tr[1, :], ch3_locs_tr[2, :], 'o', alpha=0.3, label='Channel 2 - transformed')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('y (nm)')
            ax.set_ylabel('z (nm)')
            ax.legend()

            ax = fig.add_subplot(246)
            ax.plot(ch1_locs[0, :], ch1_locs[1, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(ch3_locs_tr[0, :], ch3_locs_tr[1, :], 'o', alpha=0.3, label='Channel 2 - transformed')
            ax.set_aspect('equal')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('y (nm)')
            ax.legend()

            ax = fig.add_subplot(247)
            ax.plot(ch1_locs[0, :], ch1_locs[2, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(ch3_locs_tr[0, :], ch3_locs_tr[2, :], 'o', alpha=0.3, label='Channel 2 - transformed')
            ax.set_aspect('equal')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('z (nm)')
            ax.legend()

            ax = fig.add_subplot(248)
            ax.plot(ch1_locs[1, :], ch1_locs[2, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(ch3_locs_tr[1, :], ch3_locs_tr[2, :], 'o', alpha=0.3, label='Channel 2 - transformed')
            ax.set_aspect('equal')
            ax.set_xlabel('y (nm)')
            ax.set_ylabel('z (nm)')
            ax.legend()

            
            plt.tight_layout()
            
            # save into a new Picasso-formated hdf5
            
            ch1_fulltable['x'] = ch1_fulltable['x']/px_size # convert to px
            ch1_fulltable['y'] = ch1_fulltable['y']/px_size   
            
            ch3_fulltable['x'] = ch3_fulltable['x']/px_size # convert to px
            ch3_fulltable['y'] = ch3_fulltable['y']/px_size   
            
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