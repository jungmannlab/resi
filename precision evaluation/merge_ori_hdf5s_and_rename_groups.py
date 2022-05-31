import numpy as np
import glob
import os
import os.path
import itertools
import h5py
import pandas as pd
import re

import tools

'''Please copy and paste the path to the folder containing the data that should be analyzed.'''

path = r"W:\users\reinhardt\z_raw\Resi\211123_dbl-ext-20nm-6sites-R4\workflow_081221"
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
name_parts = ["MMStack_Pos0.ome_locs_render_ali_apicked_ori", "ClusterD4_50"] # strings contained in the names of the files to be merged
not_name_parts = [] # strings not contained in the names of the files to be merged
filename_merge = "R1_R3_ClusterD4_50_merged.hdf5" # desired name for the merged file



"""
This code requires the filenames to have a string "ori" followed by a number in their filename!
"""


dataframe_all = pd.DataFrame()

counter = 0
all_hdf5s_list = sorted(glob.glob(os.path.join(path, "*.hdf5")))

ori_hdf5s_list = []
for file in all_hdf5s_list: # searches all hdf5 files
    
    filename = os.path.split(file)[1]
    #print([name_part in filename for name_part in name_parts],[not_name_part not in filename for not_name_part in not_name_parts])
    if all(name_part in filename for name_part in name_parts) and all(not_name_part not in filename for not_name_part in not_name_parts):
        ori_hdf5s_list.append(file)

        
ori_hdf5s_list = sorted(ori_hdf5s_list, key=lambda x: int(re.search("ori(\d+)", x).group(1)))
print(ori_hdf5s_list)
for file in ori_hdf5s_list: # searches all hdf5 files
    
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
            
            
            match = re.search("ori(\d+)", file)
            ori = int(match.group(1))
            dataframe_all = pd.DataFrame(data)
            dataframe_all['ori'] = ori
            group_counter = dataframe_all['group'].max()
            #print("all", dataframe_all.shape)
        if counter > 1:
            #print(counter)
            f1 = h5py.File(file, 'r')
            a_group_key = list(f1.keys())[0]
            data = np.array(f1[a_group_key])
            
            match = re.search("ori(\d+)", file)
            ori = int(match.group(1))
            
            dataframe_add = pd.DataFrame(data)
            dataframe_add['group'] = dataframe_add['group'] + group_counter
            dataframe_add['ori'] = ori
            group_counter = dataframe_add['group'].max()
            dataframe_all = dataframe_all.append(dataframe_add, ignore_index = True)

#print(dataframe_all.keys())
dataframe_all = dataframe_all.sort_values(by=["ori","frame"])


filename_old = os.path.split(filename_example)[1]
filename_new = filename_merge
tools.picasso_hdf5(dataframe_all, filename_new, filename_old, path + "/")
