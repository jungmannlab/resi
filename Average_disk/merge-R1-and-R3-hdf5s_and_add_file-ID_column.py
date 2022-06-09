import numpy as np
import glob
import os
import os.path
import itertools
import h5py
import pandas as pd
from os.path import dirname as up
cwd = os.getcwd()
wdir = up(cwd)
os.chdir(wdir)
import tools

'''Please copy and paste the path to the folder containing the data that should be analyzed.'''

path = r"W:\users\reinhardt\z_raw\Resi\220315_disk-pair_R4CA\aligned\3d\average_cluster_centers"
filename0 = r"c.R1_resi_info_7_30_20_merged_avg-appl.hdf5" # round 1 file
filename1 = r"c.R3_resi_info_7_30_20_merged_avg-appl.hdf5" # round 2 file
filename_merge = "c.R1_resi_info_7_30_20_merged_avg-appl_R1,R3_merge.hdf5" # desired name for the merged file



file_id0 = 0
file_id1 = 1


file0 = os.path.join(path,filename0)
file1 = os.path.join(path,filename1)

dataframe_all = pd.DataFrame()

f0 = h5py.File(file0, 'r')
a_group_key = list(f0.keys())[0]
data0 = np.array(f0[a_group_key])
df0 = pd.DataFrame(data0)
# Save a file id to later seperate the hdf5 file again into both rounds
df0['file_ID'] = file_id0
# We will temporarly replace the frame column with the Origami_ID column.
# One frame will thus correspond to one origami 
# This allows to use undrift from picked such that the relative position of 
# sites from the same origami is maintained while aligning origamis instead
# of frames.
# Note that locs from different channels of the same origami have the same 
# Origami_ID and are thus assigned as the same "frame". 
# This is important because otherwise the relative orientation of both channels
# would be changed when using undrift from picked.
df0['frame_backup'] = df0['frame']
df0['frame'] = df0['Origami_ID']

f1 = h5py.File(file1, 'r')
a_group_key = list(f1.keys())[0]
data1 = np.array(f1[a_group_key])
df1 = pd.DataFrame(data1)
df1['file_ID'] = file_id1
df1['frame_backup'] = df1['frame']
df1['frame'] = df1['Origami_ID']


df12 = df0.append(df1, ignore_index = True)


tools.picasso_hdf5(df12, filename_merge, filename0, path + "/")
