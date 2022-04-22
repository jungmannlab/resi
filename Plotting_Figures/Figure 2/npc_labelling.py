# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:36:09 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 'W:/users/reinhardt/z_raw/Resi/220125_Nup96-GFP_4c/aligned/'
fname_noncluster = 'R1_7nt_150pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked.hdf5'
fname_cluster = 'R1_7nt_150pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0.hdf5'

my_df_noncluster = pd.read_hdf(path + fname_noncluster, key='locs')
my_df_cluster = pd.read_hdf(path + fname_cluster, key='locs')  

px_size = 130 # nm
my_df_noncluster.x = my_df_noncluster.x * px_size
my_df_cluster.y = my_df_cluster.y * px_size

locs_nc_x = my_df_noncluster.x
locs_nc_y = my_df_noncluster.y

locs_c_x = my_df_cluster.x
locs_c_y = my_df_cluster.y

fig, ax = plt.subplots()
# ax.scatter(locs_nc_x, locs_nc_y)
ax.scatter(locs_c_x, locs_c_y)