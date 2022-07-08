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

import tools

#Read in HDF5-File




def divide_hdf5_by_group_f(filename):
    # filename includes the path to the file and the filename itself.
    
    f1 = h5py.File(filename, 'r')
    a_group_key = list(f1.keys())[0]
    data = np.array(f1[a_group_key])

    df = pd.DataFrame(data)

    if "group" not in list(df.keys()):
        raise Exception("The file does not have a column 'group'")

    grouped = df.groupby("group")

    for name, df_group in grouped:
        path = os.path.split(filename)[0] + "/"
        filename_old = os.path.split(filename)[1]
        filename_new = '%s_ori%d.hdf5' % (filename_old[:-5],df_group['group'].mean())
        tools.picasso_hdf5(df_group, filename_new, filename_old, path)


try:
    filename = sys.argv[1]
    divide_hdf5_by_group_f(filename)
except:
    pass
