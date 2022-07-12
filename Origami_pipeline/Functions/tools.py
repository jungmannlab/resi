#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Luciano A. Masullo, Susanne Reinhardt

Collection of functions used by several scripts.
"""

import numpy as np
import h5py



def angle(p1x, p1y, p2x, p2y):
    """
    This function calculates the orientation of two coordinates p1 and p2 with 
    respect to a horicontal line going through point 1. For this a third point 
    p3 is defined as p3 = (p1x+1, p1y). Then, the angle between p3, p1 and p2 
    can be calculated. This calculation is performed for a set of p1 
    coordinates and a set of p2 coordinates.  

    Parameters
    ----------
    p1x : array
        X coordinates of the first set of points.
    p1y : array
        Y coordinates of the first set of points.
    p2x : array
        X coordinates of the second set of points.
    p2y : array
        Z coordinates of the second set of points.

    Returns
    -------
    angle : array
        The angle between each p1 p2 pair.

    """
    angle = np.zeros(len(p1x))
    for i in range(len(p1x)):
        p1 = np.array([p1x[i], p1y[i]])
        p2 = np.array([p2x[i], p2y[i]])

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
    
    

def rigid_transform_3D(A, B):
    """
    source: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    Given two sets of 3D points and their correspondence the algorithm will return a least square optimal 
    rigid transform (also known as Euclidean) between the two sets. The transform solves for 3D rotation and 3D translation, no scaling.


    rigid transformation in 3D: R A + t = B

    Input: expects 3xN matrix of points
    Returns R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector
    """
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean row wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def rigid_transform_2D(A, B):
    
    """
    source: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    Given two sets of 2D points and their correspondence the algorithm will return a least square optimal 
    rigid transform (also known as Euclidean) between the two sets. The transform solves for 2D rotation and 2D translation, no scaling.


    rigid transformation in 3D: R A + t = B

    Input: expects 3xN matrix of points
    Returns R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector
    """
    
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 2:
        raise Exception(f"matrix A is not 2xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 2:
        raise Exception(f"matrix B is not 2xN, it is {num_rows}x{num_cols}")

    # find mean row wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t
