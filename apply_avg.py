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

'''Read in Files'''

# file1
# dataset 1, not transformed (A)

filename1A = sys.argv[1]
f1A = h5py.File(filename1A, 'r')
a_group_key = list(f1A.keys())[0]
data1A = np.array(f1A[a_group_key])
print(a_group_key)


df1A = pd.DataFrame(data1A)
"""
print(type(df1A))
print(df1A.head())
print()
print(df1A.keys())
"""
grouped1A = df1A.groupby("group")


# file2
# dataset 1, transformed (average) (B)

filename1B = sys.argv[2]
f1B = h5py.File(filename1B, 'r')
a_group_key = list(f1B.keys())[0]

data1B = np.array(f1B[a_group_key])

df1B = pd.DataFrame(data1B)
grouped1B = df1B.groupby("group")



# File 3 - File to be averaged in another color channel!
# dataset 2, not tranformed (A)

filename2A = sys.argv[3]
f2A = h5py.File(filename2A, 'r')
a_group_key = list(f2A.keys())[0]
data2A = np.array(f2A[a_group_key])

df2A = pd.DataFrame(data2A)
grouped2A = df2A.groupby("group")





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


for name, grouped1A_group in grouped1A:
    # name: the grouped1A dataframe was created by grouping by the group column corresponding to the pick identifier. 
    #       name is thus the integer representing the current group in the loop.
    # grouped1A_group is a dataframe containing only one group (with identifier name) of the complete grouped1A dataframe.
    print(name)
    #print(grouped1A_group.head())
    
    grouped1A_group_xy = np.transpose(np.array(grouped1A_group[['x','y']]))
    grouped1B_group_xy = np.transpose(np.array(grouped1B[['x','y']].get_group(name)))
    R,t = rigid_transform_2D(grouped1A_group_xy, grouped1B_group_xy)
    print("R, t", R, t)
    grouped2A_group_xy = np.transpose(np.array(grouped2A[['x','y']].get_group(name)))

    grouped2B_group_xy = R @ grouped2A_group_xy + t

    print(grouped2A_group_xy)
    print(grouped2B_group_xy)
    print()
    
