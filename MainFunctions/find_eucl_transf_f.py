#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:16:53 2021

@author: Luciano A. Masullo

Find the euclidean transform that optimizes the distance between each channel
in each origami according to the alignment sites (e.g. R4 sequence)

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
import pandas as pd
import os


def find_eucl_transf_f(folder_path, ch1_files, ch3_files):
    plt.close('all')


    plt.close('all')

    px_size = 130 # nm

    # I will avoid calling them "R1" or "R3" since the data I am loading will be 
    # from R4 imagers imaged in each round

    eucl_tr_params = [] # initialize the list of Euclidean transf parameters

    for i, (file1, file3) in enumerate(zip(ch1_files, ch3_files)):
        
        print(i)
        
        # load the localizations for each channel

        ch1_fulltable = pd.read_hdf(file1, key = 'locs')
        ch1_fulltable['x'] = ch1_fulltable['x']*px_size # convert to nm
        ch1_fulltable['y'] = ch1_fulltable['y']*px_size
        
        ch3_fulltable = pd.read_hdf(file3, key = 'locs')
        ch3_fulltable['x'] = ch3_fulltable['x']*px_size # convert to nm
        ch3_fulltable['y'] = ch3_fulltable['y']*px_size
        
        number_of_R4_sites = ch1_fulltable['group'].max()+1 # usually 6 but we discard the ones lacking one of the channel
        
        locsx_ch1 = [] # lists that will have all the cleaned-up localizations for each channel
        locsy_ch1 = []
        locsx_ch3 = []
        locsy_ch3 = []
        
        for j in range(number_of_R4_sites): # loop over the localization of each site
            
            locs_ch1 = ch1_fulltable.loc[ch1_fulltable['group'] == j]
            locs_ch3 = ch3_fulltable.loc[ch3_fulltable['group'] == j]
            
            x_ch1 = locs_ch1['x'].to_numpy()
            y_ch1 = locs_ch1['y'].to_numpy()
            
            x_ch3 = locs_ch3['x'].to_numpy()
            y_ch3 = locs_ch3['y'].to_numpy()
            
            # the eucl tr estimation needs arrays with equal number of elements
            # therefore we will match the (in general different) number of 
            # elements in ch1 and ch3
            
            if np.size(locs_ch1) > np.size(locs_ch3):
            
                size = np.shape(locs_ch3)[0]
                x_ch1 = np.random.choice(x_ch1, size=size, replace=False, p=None)
                y_ch1 = np.random.choice(y_ch1, size=size, replace=False, p=None)
                
            elif np.size(locs_ch1) < np.size(locs_ch3):
                
                size = np.shape(locs_ch1)[0]
                x_ch3 = np.random.choice(x_ch3, size=size, replace=False, p=None)
                y_ch3 = np.random.choice(y_ch3, size=size, replace=False, p=None)
                   
            else:
                
                pass
            
            locsx_ch1.append(x_ch1)
            locsy_ch1.append(y_ch1)
            locsx_ch3.append(x_ch3)
            locsy_ch3.append(y_ch3)
         
        # put together all localizations for each channel    
        locsx_ch1 = np.concatenate(locsx_ch1, axis=0)
        locsy_ch1 = np.concatenate(locsy_ch1, axis=0)
        locsx_ch3 = np.concatenate(locsx_ch3, axis=0)
        locsy_ch3 = np.concatenate(locsy_ch3, axis=0)
        
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
        params = transf_estimate.rotation, transf_estimate.translation
        
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
            print ("eucl_trasnf folder already exists")
        
        plt.savefig(folder_path + '/eucl_transf/' + 'origami' + str(i) + '.pdf', format='pdf')

        # save the params in the list
        eucl_tr_params.append(params)
       
    # save the final eucl_tr_params list

    angle = []
    tx = []
    ty = []
    for i in range(len(eucl_tr_params)):
        
        angle.append(eucl_tr_params[i][0])
        tx.append(eucl_tr_params[i][1][0])
        ty.append(eucl_tr_params[i][1][1])
        
    d = {'rotation':angle, 'translation x':tx, 'translation y':ty}
    df = pd.DataFrame(d)
    df.to_csv(folder_path +  r'/eucl_transf/'+ 'eucl_transf_' + 'data.csv', sep='\t', encoding='utf-8')
    df.to_excel(folder_path + r'/eucl_transf/' + 'eucl_transf_' + 'data.xlsx') 

            

    #TODO: save plots

