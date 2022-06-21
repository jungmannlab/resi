# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:29:31 2021

@author: reinhardt
"""

import h5py
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
import h5py as _h5py
import numpy as _np
from os.path import dirname as up
cwd = os.getcwd()
wdir = up(cwd)
os.chdir(wdir)
import tools

#Read in HDF5-File

path = r"W:\users\reinhardt\z_raw\Resi\220315_disk-pair_R4CA\aligned\3d\average_cluster_centers"
filename_input = "c.R1_resi_info_7_30_20_merged_avg-appl_R1,R3_merge_render.hdf5" # desired name for the merged file



def divide_hdf5_by_fileID_f(path, filename):
    # filename includes the path to the file and the filename itself.
    file = os.path.join(path,filename)
    f1 = h5py.File(file, 'r')
    a_group_key = list(f1.keys())[0]
    data = np.array(f1[a_group_key])

    df = pd.DataFrame(data)

    if "file_ID" not in list(df.keys()):
        raise Exception("The file does not have a column 'file_ID'")

    df['frame'] = df['frame_backup']
    grouped = df.groupby("file_ID")

    for name, df_group in grouped:
        filename_new = '%s_file%d.hdf5' % (filename[:-5],df_group['file_ID'].mean())
        tools.picasso_hdf5(df_group, filename_new, filename, path + "/")


divide_hdf5_by_fileID_f(path, filename_input)
