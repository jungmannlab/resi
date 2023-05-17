#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Luciano A. Masullo, Susanne Reinhardt

For each origami the best euclidian transformation to align round 1 and 2 are 
found based on the picked alignment sites ("R4"). 
Input: The R4 alignment sites have to be picked manually
       for each pair of oriN.hdf5 files in Picasso Render and saved in a 
       subfolder callded 'alignment_picks'.
Output: saved in 'eucl_transf' subfolder
- csv and excel file containing the rotation and translation parameters
- pdf images of each origami's alignment sites showing the overlay of both 
  rounds before and after the alignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
import pandas as pd
import os
import sys
import re


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Functions import tools



def find_eucl_transf_f(folder_path, ch1_files, ch3_files, px_size):
    '''
    Estimates the best euclidian transformation between both channels of a
    origami based on the picked alignment sites.

    Parameters
    ----------
    folder_path : string
        Rath to the folder containing the picked alignment sites for each 
        origami and round.
    ch1_files : list of strings
        Filenames of all round 1 alignment files.
    ch3_files : list of strings
        Filenames of all round 2 alignment files.
    px_size : int
        Size of a pixel in nm.
    '''

    plt.close('all')



    eucl_tr_params = [] # initialize the list of Euclidean transf parameters

    for i, (file1, file3) in enumerate(zip(ch1_files, ch3_files)):
        
        # Check if Origami files from both channels were assigned correctly 
        # to each other.
        match1 = re.search("ori(\d+)", file1)
        match3 = re.search("ori(\d+)", file3)
        if match1 and match3:
            if match1.group(1) == match3.group(1):
                pick = match1.group(1)

            else:
                raise Exception('Origamis from channel 1 and channel 3 were not assigned correctly to each other.')
        else:
            raise Exception('Origamis from channel 1 and channel 3 cannot be assigned to each other. Check if "ori" followed by a number is included in the filenames.')
      
        
        # Load the localizations for each channel
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
            raise Exception('The alignment picks in both channels have to be either 2D or 3D data.')


        number_of_R4_sites = ch1_fulltable['group'].max()+1 
        
        
        # the eucl tr estimation needs arrays with equal number of elements
        # therefore we will match the (in general different) number of 
        # elements in ch1 and ch3
        locsx_ch1 = [] # lists that will have all the cleaned-up localizations for each channel
        locsy_ch1 = []
        locsx_ch3 = []
        locsy_ch3 = []
        if flag_3D:
            locsz_ch1 = []
            locsz_ch3 = []
        
        for j in range(number_of_R4_sites): # loop over the localization of each site
            
            locs_ch1 = ch1_fulltable.loc[ch1_fulltable['group'] == j]
            locs_ch3 = ch3_fulltable.loc[ch3_fulltable['group'] == j]
            
            x_ch1 = locs_ch1['x'].to_numpy()
            y_ch1 = locs_ch1['y'].to_numpy()
            if flag_3D:
                z_ch1 = locs_ch1['z'].to_numpy()
        
            x_ch3 = locs_ch3['x'].to_numpy()
            y_ch3 = locs_ch3['y'].to_numpy()
            if flag_3D:
                z_ch3 = locs_ch3['z'].to_numpy()
            
            
            if np.size(locs_ch1) > np.size(locs_ch3):
            
                size = np.shape(locs_ch3)[0]
                x_ch1 = np.random.choice(x_ch1, size=size, replace=False, p=None)
                y_ch1 = np.random.choice(y_ch1, size=size, replace=False, p=None)
                if flag_3D:
                    z_ch1 = np.random.choice(z_ch1, size=size, replace=False, p=None)
                
            elif np.size(locs_ch1) < np.size(locs_ch3):
                
                size = np.shape(locs_ch1)[0]
                x_ch3 = np.random.choice(x_ch3, size=size, replace=False, p=None)
                y_ch3 = np.random.choice(y_ch3, size=size, replace=False, p=None)
                if flag_3D:
                    z_ch3 = np.random.choice(z_ch3, size=size, replace=False, p=None)
            
            locsx_ch1.append(x_ch1)
            locsy_ch1.append(y_ch1)
            locsx_ch3.append(x_ch3)
            locsy_ch3.append(y_ch3)
            if flag_3D:
                locsz_ch1.append(z_ch1)
                locsz_ch3.append(z_ch3)

         
        # put together all localizations for each channel    
        locsx_ch1 = np.concatenate(locsx_ch1, axis=0)
        locsy_ch1 = np.concatenate(locsy_ch1, axis=0)
        locsx_ch3 = np.concatenate(locsx_ch3, axis=0)
        locsy_ch3 = np.concatenate(locsy_ch3, axis=0)
        if flag_3D:
            locsz_ch1 = np.concatenate(locsz_ch1, axis=0)
            locsz_ch3 = np.concatenate(locsz_ch3, axis=0)
        
        
        # Find the best rotation matrix and translation vector to align the 
        # R4 alignment sites of both rounds
        if not flag_3D: # 2d data

            # prepare the data in the matrix form
            N = int(len(locsx_ch1))
            A = np.concatenate((locsx_ch1, locsy_ch1)).reshape(2, N).T
            B = np.concatenate((locsx_ch3, locsy_ch3)).reshape(2, N).T
            
            # plot the data before eucl transf
            fig, ax = plt.subplots(2, figsize=(12,12))
            
            ax[0].plot(A[:, 0], A[:, 1], 'o', alpha=0.3, label='Channel 1')
            ax[0].plot(B[:, 0], B[:, 1], 'o', alpha=0.3, label='Channel 2')
            ax[0].set_aspect('equal')
            ax[0].set_xlabel('x (nm)')
            ax[0].set_ylabel('y (nm)')
            ax[0].legend()
                
            # estimate best eucl transf
            transf_estimate = tf.EuclideanTransform()
            transf_estimate.estimate(B, A)
            params = pick, transf_estimate.rotation, transf_estimate.translation
            
            # transform the data with the opt params    
            data_est = transf_estimate(B)

            # plot the data after the transformation
            ax[1].plot(A[:, 0], A[:, 1], 'o', alpha=0.3, label='Channel 1')
            ax[1].plot(data_est[:, 0], data_est[:, 1], 'o', alpha=0.3, label='Channel 2 - transformed')
            ax[1].set_aspect('equal')
            ax[1].set_xlabel('x (nm)')
            ax[1].set_ylabel('y (nm)')
            ax[1].legend()
            
            plt.tight_layout()
            
            try:
                os.mkdir(folder_path + '/eucl_transf')
            
            except OSError:
                #print ("eucl_trasnf folder already exists")
                pass
                  
            # save the figure showing the alignment
            plt.savefig(folder_path + '/eucl_transf/' + 'origami' + str(pick) + '.pdf', format='pdf')

            # save the params in the list
            eucl_tr_params.append(params)
           
            

        else: # 3d data

            # prepare the data in the matrix form (the function rigid_transform_3D(A,B) expects the input arrays as 3xN arrays)
            N = int(len(locsx_ch1))
            A = np.concatenate((locsx_ch1, locsy_ch1, locsz_ch1)).reshape(3, N)
            B = np.concatenate((locsx_ch3, locsy_ch3, locsz_ch3)).reshape(3, N)

            # plot the data before eucl transf
            fig = plt.figure(figsize=(24,12))


            ax = fig.add_subplot(241, projection = '3d')
            ax.plot(A[0, :], A[1, :], A[2, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(B[0, :], B[1, :], B[2, :], 'o', alpha=0.3, label='Channel 2')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('y (nm)')
            ax.set_ylabel('z (nm)')
            ax.legend()

            ax = fig.add_subplot(242)
            ax.plot(A[0, :], A[1, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(B[0, :], B[1, :], 'o', alpha=0.3, label='Channel 2')
            ax.set_aspect('equal')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('y (nm)')
            ax.legend()

            ax = fig.add_subplot(243)
            ax.plot(A[0, :], A[2, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(B[0, :], B[2, :], 'o', alpha=0.3, label='Channel 2')
            ax.set_aspect('equal')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('z (nm)')
            ax.legend()

            ax = fig.add_subplot(244)
            ax.plot(A[1, :], A[2, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(B[1, :], B[2, :], 'o', alpha=0.3, label='Channel 2')
            ax.set_aspect('equal')
            ax.set_xlabel('y (nm)')
            ax.set_ylabel('z (nm)')
            ax.legend()

            # estimate best eucl transf
            R,t = tools.rigid_transform_3D(A, B)
            params = pick, R, t

            # transform the data with the opt params 
            data_est = np.linalg.inv(R) @ (B - t)

            
            # plot the data after the transformation
            ax = fig.add_subplot(245, projection = '3d')
            ax.plot(A[0, :], A[1, :], A[2, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(data_est[0, :], data_est[1, :], data_est[2, :], 'o', alpha=0.3, label='Channel 2 - transformed')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('y (nm)')
            ax.set_ylabel('z (nm)')
            ax.legend()

            ax = fig.add_subplot(246)
            ax.plot(A[0, :], A[1, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(data_est[0, :], data_est[1, :], 'o', alpha=0.3, label='Channel 2 - transformed')
            ax.set_aspect('equal')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('y (nm)')
            ax.legend()

            ax = fig.add_subplot(247)
            ax.plot(A[0, :], A[2, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(data_est[0, :], data_est[2, :], 'o', alpha=0.3, label='Channel 2 - transformed')
            ax.set_aspect('equal')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('z (nm)')
            ax.legend()

            ax = fig.add_subplot(248)
            ax.plot(A[1, :], A[2, :], 'o', alpha=0.3, label='Channel 1')
            ax.plot(data_est[1, :], data_est[2, :], 'o', alpha=0.3, label='Channel 2 - transformed')
            ax.set_aspect('equal')
            ax.set_xlabel('y (nm)')
            ax.set_ylabel('z (nm)')
            ax.legend()

            
            plt.tight_layout()
            
            try:
                os.mkdir(folder_path + '/eucl_transf')
            
            except OSError:
                #print ("eucl_trasnf folder already exists")
                pass
            
            plt.savefig(folder_path + '/eucl_transf/' + 'origami' + str(pick) + '.pdf', format='pdf')

            # save the params in the list
            eucl_tr_params.append(params)
           



    # save the final eucl_tr_params list
    if not flag_3D:
        picks = []
        angle = []
        tx = []
        ty = []
        for i in range(len(eucl_tr_params)):
            picks.append(eucl_tr_params[i][0])
            angle.append(eucl_tr_params[i][1])
            tx.append(eucl_tr_params[i][2][0])
            ty.append(eucl_tr_params[i][2][1])
            
        d = {'pick':picks, 'rotation':angle, 'translation x':tx, 'translation y':ty}
        df = pd.DataFrame(d)   

    else:
        picks = []
        rotation_matrix = []
        translation_vector = []
        for i in range(len(eucl_tr_params)):
            picks.append(eucl_tr_params[i][0])
            rotation_matrix.append(eucl_tr_params[i][1])
            translation_vector.append(eucl_tr_params[i][2])
        
        d = {'pick':picks, 'rotation matrix':rotation_matrix, 'translation vector':translation_vector}
        df = pd.DataFrame(d)


    df.to_csv(folder_path +  r'/eucl_transf/'+ 'eucl_transf_' + 'data.csv', sep='\t', encoding='utf-8')
    df.to_excel(folder_path + r'/eucl_transf/' + 'eucl_transf_' + 'data.xlsx') 
