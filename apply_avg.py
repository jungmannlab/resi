# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:59:46 2021

@author: reinhardt
"""

    
"""


1) Read in 2xhdf5 picked localization Files

2) Define center of mass for each picked region for both files

3) Measure x and y offset of each center of mass per pick to the averaged pick (save x-y translation per pick)

4) Move the picked area to the average according to x and y offset

5) Rotate picked area per pick (1 loc should be enough) to overlay with the averaged image (save rotation angle)

6) Transform dataset with different color channel with correct transformations & save hdf5 file!

"""



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import numpy as np
import time
import sys
import seaborn as sns
from sklearn.cluster import DBSCAN
import math
import matplotlib.pyplot as plt
import pandas as pd
import os

import tools

'''Read in Files'''

# file1
# dataset 1, not transformed (A)

filename1A = sys.argv[1]
f1A = h5py.File(filename1A, 'r')
a_group_key = list(f1A.keys())[0]
data1A = np.array(f1A[a_group_key])

df1A = pd.DataFrame(data1A)
"""
print(type(df1A))
print(df1A.head())
print()
print(df1A.keys())
"""
grouped1A = df1A.groupby("group")

try:
    grouped1A['z']
except:
    print("Dataset 1 not averged: 2D data detected.")
    flag_3D_1A = False
else:
    print("Dataset 1 not averged: 3D data detected.")
    flag_3D_1A = True



# file2
# dataset 1, transformed (average) (B)

filename1B = sys.argv[2]
f1B = h5py.File(filename1B, 'r')
a_group_key = list(f1B.keys())[0]
data1B = np.array(f1B[a_group_key])

df1B = pd.DataFrame(data1B)
grouped1B = df1B.groupby("group")

try:
    grouped1B['z']
except:
    print("Dataset 1 averged: 2D data detected.")
    flag_3D_1B = False
else:
    print("Dataset 1 averged: 3D data detected.")
    flag_3D_1B = True


# File 3 - File to be averaged in another color channel!
# dataset 2, not tranformed (A)

filename2A = sys.argv[3]
f2A = h5py.File(filename2A, 'r')
a_group_key = list(f2A.keys())[0]
data2A = np.array(f2A[a_group_key])

df2A = pd.DataFrame(data2A)
#print(df2A['group'].head())
grouped2A = df2A.groupby("group")
#print(grouped2A.head())

try:
    grouped2A['z']
except:
    print("Dataset 2 not averged: 2D data detected.")
    flag_3D_2A = False
else:
    print("Dataset 2 not averged: 3D data detected.")
    flag_3D_2A = True


if not flag_3D_1A == flag_3D_1B:
    sys.exit("The not averaged and the averaged version of dataset 1 both need to be either 2d or 3d.")
if not flag_3D_1A == flag_3D_2A:
    sys.exit("The dataset serving as a template for averaging and the dataset to be transformed both need to be either 2d or 3d.")

flag_3D = flag_3D_1A




#source: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
#Given two sets of 3D points and their correspondence the algorithm will return a least square optimal 
# rigid transform (also known as Euclidean) between the two sets. The transform solves for 3D rotation and 3D translation, no scaling.


# rigid transformation in 3D: R A + t = B

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
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


df2B = df2A.copy()
if not flag_3D:
    list2B_xy = []
    if 'orientation' in df2B.keys():
        list2B_xy_cross = []
else:
    list2B_xyz = []
    if 'orientation' in df2B.keys():
        list2B_xyz_cross = []
    
for name, group_2A in grouped2A:
    # name: the grouped2A dataframe was created by grouping by the group column corresponding to the pick identifier. 
    #       name is thus the integer representing the current group in the loop.
    # group_2A is a dataframe containing only one group (with identifier name) of the complete grouped1A dataframe.
    
    # There are some picks that were originally picked, but then were discarded during further analysis when one had
    # a very close look at eve ry single pick. In such a case the pick with its ID is still present in grouped1A and 
    # grouped1B, but does not exist anymore in grouped2A. Therefore, the code checks only takes the picks that are
    # still present in grouped2A.

    # Reconstract transformation applied to dataset 1 on by Picasso Average  
    if not flag_3D:  
        group_1A_xy = np.transpose(np.array(grouped1A[['x','y']].get_group(name)))
        group_1B_xy = np.transpose(np.array(grouped1B[['x','y']].get_group(name)))
        R,t = rigid_transform_2D(group_1A_xy, group_1B_xy)
    else:
        group_1A_xyz = np.transpose(np.array(grouped1A[['x','y','z']].get_group(name)))
        group_1B_xyz = np.transpose(np.array(grouped1B[['x','y','z']].get_group(name)))
        R,t = rigid_transform_3D(group_1A_xyz, group_1B_xyz)


    # Apply transformation to dataset 2
    if not flag_3D:
        group_2A_xy = np.transpose(np.array(group_2A[['x','y']]))
        group_2B_xy = R @ group_2A_xy + t
        list2B_xy.extend(list(np.transpose(group_2B_xy)))
    else:
        group_2A_xyz = np.transpose(np.array(group_2A[['x','y','z']]))
        group_2B_xyz = R @ group_2A_xyz + t
        list2B_xyz.extend(list(np.transpose(group_2B_xyz)))
    
    
    if 'orientation' in df2B.keys():
        if not flag_3D:
            group_2A_xy_cross = np.transpose(np.array(group_2A[['crossNND_x','crossNND_y']]))
            group_2B_xy_cross = R @ group_2A_xy_cross + t
            list2B_xy_cross.extend(list(np.transpose(group_2B_xy_cross)))
        else:
            group_2A_xyz_cross = np.transpose(np.array(group_2A[['crossNND_x','crossNND_y','crossNND_z']]))
            group_2B_xyz_cross = R @ group_2A_xyz_cross + t
            list2B_xyz_cross.extend(list(np.transpose(group_2B_xyz_cross)))
        
        

if not flag_3D:
    df2B_xy = np.array(list2B_xy)
    df2B['x'] = df2B_xy[:,0]
    df2B['y'] = df2B_xy[:,1]

    if 'orientation' in df2B.keys():
        df2B_xy_cross = np.array(list2B_xy_cross)
        df2B['crossNND_x'] = df2B_xy_cross[:,0]
        df2B['crossNND_y'] = df2B_xy_cross[:,1]

        df2B['orientation_avg'] = tools.angle(df2B['x'], df2B['y'], df2B['crossNND_x'], df2B['crossNND_y'])

else:
    
    df2B_xyz = np.array(list2B_xyz)
    df2B['x'] = df2B_xyz[:,0]
    df2B['y'] = df2B_xyz[:,1]
    df2B['z'] = df2B_xyz[:,2]
    if 'orientation' in df2B.keys():
        df2B_xyz_cross = np.array(list2B_xyz_cross)
        df2B['crossNND_x'] = df2B_xyz_cross[:,0]
        df2B['crossNND_y'] = df2B_xyz_cross[:,1]
        df2B['crossNND_z'] = df2B_xyz_cross[:,2]

        df2B['orientation_avg'] = tools.angle(df2B['x'], df2B['y'], df2B['crossNND_x'], df2B['crossNND_y'])
        # NOTE: This is only the orientation in xy direction and does not consider the angle in z direction!
    
    
df2B['Origami_ID'] = df2B['group']

df2B_filename = os.path.split(filename2A)[1]
df2B_filename = df2B_filename[:-5] + "_avg-appl.hdf5"
tools.picasso_hdf5(df2B, df2B_filename, os.path.split(filename2A)[1], os.path.split(filename2A)[0] + "/")