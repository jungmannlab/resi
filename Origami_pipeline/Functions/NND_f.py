# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 12:28:46 2022

@author: reinhardt Rafal
"""
import numpy as _np
import numpy as np
import pandas as pd
import sys
import os
from sklearn.neighbors import NearestNeighbors as NN

from Functions import tools

def nn_analysis(
    x1, x2, 
    y1, y2, 
    z1, z2,
    nn_count, 
    same_channel, 
):
    '''
    

    Parameters
    ----------
    x1 : pandas series
    x2 : pandas series
    y1 : pandas series
    y2 : pandas series
    z1 : pandas series
    z2 : pandas series
        Coordinates of two datasets.
    nn_count : int
        Input value for sklearn's NearestNeighbors function. Sets up until
        which order higher nearest neighbors should be calculated,
        f.ex. 1st, 2nd or until 3rd NND
    same_channel : bool
        if True, both datasets are equal and the NND distribution is calculated
        within one dataset.
        if False, both datasets differ and the nearest neighbor of each 
        coordinate in dataset 2 is searched in dataset 1.

    Returns
    -------
    nnd : list
        Nearest Neighbor Distance. Optionally higher NNDs in further columns.
    nn_indices : list
        Index of nearest neighbors in dataset 1 to dataset 2 coordinates.
    nn_partner : np array
        Coordinates of nearest neighbors in dataset 1 to dataset 2 coordinates.

    '''
    if z1 is not None: # 3D
        input1 = _np.stack((x1, y1, z1)).T
        input2 = _np.stack((x2, y2, z2)).T
    else: # 2D
        input1 = _np.stack((x1, y1)).T
        input2 = _np.stack((x2, y2)).T
    if same_channel:
        model = NN(n_neighbors=nn_count+1)
    else:
        model = NN(n_neighbors=nn_count)
    model.fit(input1)
    nnd, indices = model.kneighbors(input2)
    print('indices', indices)
    if same_channel:
        nnd = nnd[:, 1:] # ignore the zero distance
        # Get indices for first NND partners
        nn_indices = [sublist[1] for sublist in indices]
    else:
        nn_indices = [sublist[0] for sublist in indices]
    #print('nn_indices', nn_indices)
    #print('input2', input2)
    print('input1', input1)
    print('input2', input2)
    print('nn_indices', nn_indices)
    nn_partner = np.array([input1[nn_indices[i]] for i in range(len(input2))])
    print('nn_partner', nn_partner)
    
    return nnd, nn_indices, nn_partner


def nearest_neighbor(protein1, file_resi1, protein2, file_resi2, px_size):
    '''
    Initiates nearest neighbor calculations betweend two datasets than can be
    equal or different.
    Saves calculated distances in .csv format. In case of two different 
    datasets it adds columns to the resi hdf5 files with the coordinates 
    of the nearest neighbor in the other file and the distance to it.

    Parameters
    ----------
    protein1 : TYPE
        DESCRIPTION.
    file_resi1 : TYPE
        DESCRIPTION.
    protein2 : TYPE
        DESCRIPTION.
    file_resi2 : TYPE
        DESCRIPTION.
    px_size : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    """
    
    Parameters
    ----------
    channel1 : int
        Channel to calculate nearest neighbors distances
    channel2 : int
        Second channel to calculate nearest neighbor distances
    """

    'Load from Resi.hdf5 not form npz'
    'Flag if same channel or not'


    # NND within channel or crossNND betwen channels?
    # used for avoiding zero distances (to self)
    same_channel = file_resi1 == file_resi2

    """load Resi files"""
    
    df_resi1 = pd.read_hdf(file_resi1, key = 'locs') 
    x1 = df_resi1['x']*px_size  # Input is in px
    y1 = df_resi1['y']*px_size
    try:
        z1 = df_resi1['z'] # Input is in nm
        flag_3D_1 = True
    except:
        z1 = None
        flag_3D_1 = False    
    
    
    if not same_channel: # load second channel
        df_resi2 = pd.read_hdf(file_resi2, key = 'locs')
        x2 = df_resi2['x']*px_size  # Input is in px
        y2 = df_resi2['y']*px_size
        try:
            z2 = df_resi2['z'] # Input is in nm
            flag_3D_2 = True
        except:
            z2 = None
            flag_3D_2 = False  

        if not flag_3D_1 == flag_3D_2:
            sys.exit("Both datasets have to be either 2D or 3D.")
    else: # same channel
        x2 = x1
        y2 = y1
        if flag_3D_1:
            z2 = z1
        
        
    flag_3D = flag_3D_1

    if same_channel:
        nn_count = 1 # only first nearest neighbor 
        # The first NN will be calculated to itself, so we will later discard
        # the first column and take only the second column corresponding to the
        # the conventioanl first neareast neighbor.
    else:
        nn_count = 1 # only first cross nearest neighbor
    
    nnd, nn_indices, nn_partner = nn_analysis(
        x1, x2, 
        y1, y2, 
        z1, z2,
        nn_count,  
        same_channel, 
    )

    # Create subfolder for nearest neighbor csv files
    try:
        os.mkdir(os.path.split(file_resi1)[0] + '/AdditionalOutputs')
    except OSError:
        pass
    path =  os.path.split(file_resi1)[0] + "/AdditionalOutputs/"
    fname = path + os.path.split(file_resi1)[1]
    
    # save nn distances as csv
    if same_channel:
        np.savetxt('%s_nn_%s.csv' %(fname[:-5], protein1), nnd)
    else:
        np.savetxt('%s_nn_%s_to_%s.csv' %(fname[:-5], protein1, protein2), nnd)
        

    # Extend the content of file_resi_2 with columns containing the coordinates 
    # to its nearest neighbors in file_resi1 as well as their distance and 
    # orientation in the xy plane to each other

    if not same_channel:
        print(df_resi2['x'])
        #print(nn_indices)
        df_resi2['crossNND_ID'] = nn_indices
        df_resi2['crossNND_x'] = nn_partner[:,0]/px_size # Back from nm to pix
        df_resi2['crossNND_y'] = nn_partner[:,1]/px_size
        if flag_3D:
            df_resi2['crossNND_z'] = nn_partner[:,2]/px_size # save this in pix
        df_resi2['crossNND'] = nnd
        if flag_3D:
            df_resi2['crossNND_dxy'] = np.sqrt((df_resi2['x']-df_resi2['crossNND_x'])**2+
                                                 (df_resi2['y']-df_resi2['crossNND_y'])**2)*px_size
            df_resi2['crossNND_dz'] = (df_resi2['z']-df_resi2['crossNND_z']*px_size)
        df_resi2['orientation'] = tools.angle(df_resi2['x'], df_resi2['y'], df_resi2['crossNND_x'], df_resi2['crossNND_y'])
    
    
        path = os.path.split(file_resi2)[0] + "/"
        filename_old = os.path.split(file_resi2)[1]
        filename_new = '%s_info.hdf5' % (filename_old[:-5])
        tools.picasso_hdf5(df_resi2, filename_new, filename_old, path)
    


