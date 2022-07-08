#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: reinhardt

This script splits a Picasso hdf5 file into several seperate hdf5 files. 
The localizations are assigned to a new file depending on their ID in the 
'group' column of the hdf5 file.
Thus we obtain one file per pick ( = per origami) made in Picasso Render 
named by its group number: _ori0.hdf5, _ori1.hdf5 ....
"""




'''
=============================================================================
User Input
=============================================================================
Specify the filepath as well as the filename of the hdf5 file. 
'''

path = r"W:\users\reinhardt\z.software\Git\RESI\RESI\Origami_pipeline\TestData"
filename = r"R3_filter_apicked.hdf5"





'''
=============================================================================
Script - No need for modifications
=============================================================================
'''

import h5py
import numpy as np
import pandas as pd
import os
from Functions.tools import picasso_hdf5


def divide_hdf5_by_group_f(file):
    """
    This function takes a Picasso hdf5 file containing a group column and 
    saves the localizations of each group into a seperate hdf5 file. 

    Parameters
    ----------
    file : string
        file is the full path to a Picasso hdf5 file that will be split up.

    Raises
    ------
    Exception
        If file does not have a 'group' column the file cannot be split up 
        into seperate hdf5 files.
    """

    f1 = h5py.File(file, 'r')
    a_group_key = list(f1.keys())[0]
    data = np.array(f1[a_group_key])

    df = pd.DataFrame(data)

    if "group" not in list(df.keys()):
        raise Exception("The file does not have a column 'group'. Did you save picked regions in Picasso?")

    grouped = df.groupby("group")

    for name, df_group in grouped:
        path = os.path.split(file)[0] + "/"
        filename_old = os.path.split(file)[1]
        filename_new = '%s_ori%d.hdf5' % (filename_old[:-5],df_group['group'].mean())
        picasso_hdf5(df_group, filename_new, filename_old, path)

file = os.path.join(path,filename)

divide_hdf5_by_group_f(file)

