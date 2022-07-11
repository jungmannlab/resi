import numpy as np
import glob
import os
import os.path
import itertools
import h5py
import pandas as pd

import tools

'''Please copy and paste the path to the folder containing the data that should be analyzed.'''

path = r"W:\users\reinhardt\z_raw\Resi\220315_disk-pair_R4CA\aligned\3d"
'''
name_parts = ["resi", "R1", "info", ] # strings contained in the names of the files to be merged
not_name_parts = ["full", "merge", "avg"] # strings not contained in the names of the files to be merged
filename_merge = "z.test.R1_resi_info_merged.hdf5" # desired name for the merged file
'''
'''
name_parts = ["resi_7_30_20", "R3", "info", ] # strings contained in the names of the files to be merged
not_name_parts = ["full", "merge", "avg"] # strings not contained in the names of the files to be merged
filename_merge = "c.R3_resi_info_7_30_20_merged.hdf5" # desired name for the merged file
'''
name_parts = ["ClusterD", "R3"] # strings contained in the names of the files to be merged
not_name_parts = ["full", "merge", "avg"] # strings not contained in the names of the files to be merged
filename_merge = "c.R3_ClusterD7_30_20_merged.hdf5" # desired name for the merged file


dataframe_all = pd.DataFrame()

counter = 0
for file in glob.glob(os.path.join(path, "*.hdf5")): # searches all hdf5 files
    
    filename = os.path.split(file)[1]
    #print([name_part in filename for name_part in name_parts],[not_name_part not in filename for not_name_part in not_name_parts])
    if all(name_part in filename for name_part in name_parts) and all(not_name_part not in filename for not_name_part in not_name_parts):
        #print(file)
        counter += 1
        if counter == 1:
            #print(counter)
            filename_example = file
            
            f1 = h5py.File(file, 'r')
            a_group_key = list(f1.keys())[0]
            data = np.array(f1[a_group_key])
            
            dataframe_all = pd.DataFrame(data)
            #print("all", dataframe_all.shape)
        if counter > 1:
            #print(counter)
            f1 = h5py.File(file, 'r')
            a_group_key = list(f1.keys())[0]
            data = np.array(f1[a_group_key])
            
            dataframe_add = pd.DataFrame(data)
            dataframe_all = dataframe_all.append(dataframe_add, ignore_index = True)

#print(dataframe_all.keys())
dataframe_all = dataframe_all.sort_values(by=["group","frame"])


filename_old = os.path.split(filename_example)[1]
filename_new = filename_merge
tools.picasso_hdf5(dataframe_all, filename_new, filename_old, path + "/")
