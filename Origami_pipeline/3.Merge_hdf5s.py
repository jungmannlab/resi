#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Susanne Reinhardt

This file allows to merge an arbitrary amount of hdf5 files in a folder. 
The user can specify words or expressions that have to occur in the filename 
to identify the files that should be merged.
There might be files that do share these common expressions, but should not be
merged with the other files. These can be excluded by specifing words that 
must not be part of the filename to be integrated into the merged file.
Besides the merged hdf5 file a csv table is saved that lists all the files that
went into the average. This can be used to verify if the correct files were
merged.

"""


'''
=============================================================================
User Input
=============================================================================
'''

'''Path to the folder containing the hdf5 files to be merged.'''
path = r"W:\users\reinhardt\z.software\Git\RESI\RESI\Origami_pipeline\3D_test_data_disk_result"

'''Specify which expressions have to be part of a filename to be integrated into the merged file.'''
name_parts = ["R3", "resi_7_30_20", "info"]

'''Specify which expressions must NOT be part of a filename to be integrated into the merged file.'''
'''If running the script several times it happens easily that a previously generated merge file 
   is acccidentally included. Thus it is recommended to always add an expression to exclude merged files'''
not_name_parts = ["merge"] 

'''Name of the merged file'''
filename_merge = "R3_resi_7_30_20_info_merge.hdf5"



'''
=============================================================================
Script - No need for modifications
=============================================================================
'''


import numpy as np
import glob
import sys
import os
import os.path
import h5py
import pandas as pd

from Functions.tools import picasso_hdf5


dataframe_all = pd.DataFrame()

counter = 0
merged_files = []
# generate a list of all hdf5 files in the specified folder
for file in glob.glob(os.path.join(path, "*.hdf5")): 
    
    filename = os.path.split(file)[1]
    # Check if the required name parts are in the filename. 
    # If additionally forbidden name parts are in the filename, this file
    # will not be merged together with the files fulfilling the requirements.
    if all(name_part in filename for name_part in name_parts) and all(not_name_part not in filename for not_name_part in not_name_parts):
        counter += 1
        merged_files.append(filename)
        # When the first file that fulfills the criteria is found a dataframe
        # is created. Later found files will be appended to this dataframe.
        if counter == 1:
            filename_example = file
            
            f1 = h5py.File(file, 'r')
            a_group_key = list(f1.keys())[0]
            data = np.array(f1[a_group_key])
            
            dataframe_all = pd.DataFrame(data)
        if counter > 1:
            f1 = h5py.File(file, 'r')
            a_group_key = list(f1.keys())[0]
            data = np.array(f1[a_group_key])
            
            dataframe_add = pd.DataFrame(data)
            dataframe_all = dataframe_all.append(dataframe_add, ignore_index = True)

if merged_files == []:
    sys.exit("No file fullfilled the criteria. No merged file was created.")

# Sort the rows by their group ID and the frame number
dataframe_all = dataframe_all.sort_values(by=["group","frame"])

# Save the merged hdf5 file
filename_old = os.path.split(filename_example)[1]
filename_new = filename_merge
picasso_hdf5(dataframe_all, filename_new, filename_old, path + "/")

# Save a txt file with all the lists that were incorporated into the merged file
np.savetxt('%s.csv' %(os.path.join(path, filename_new)[:-5]), merged_files, header = 'Merged files', fmt = '%s')
