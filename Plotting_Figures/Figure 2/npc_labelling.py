# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:36:09 2022

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

path = 'W:/users/reinhardt/z_raw/Resi/220125_Nup96-GFP_4c/aligned/'

fname_noncluster = {}
fname_cluster = {}

fname_noncluster['R1'] = 'R1_7nt_150pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked.hdf5'
fname_cluster['R1'] = 'R1_7nt_150pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0.hdf5'

fname_noncluster['R2'] = 'R2_7nt_100pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked.hdf5'
fname_cluster['R2'] = 'R2_7nt_100pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0.hdf5'

fname_noncluster['R3'] = 'R3_7nt_100pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked.hdf5'
fname_cluster['R3'] = 'R3_7nt_100pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0.hdf5'

fname_noncluster['R4'] = 'R4_7nt_100pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked.hdf5'
fname_cluster['R4'] = 'R4_7nt_100pM_18mW_150ms_561_1_MMStack_Pos0.ome_locs_render_render_filter_aligned_apicked_ClusterD11_10_22.0.hdf5'

px_size = 130 # nm

locs_nc_x = {}
locs_nc_y = {}
locs_nc_z = {}

locs_c_x = {}
locs_c_y = {}
locs_c_z = {}

display_xlocs_c = {}
display_ylocs_c = {}
display_zlocs_c = {}

display_xlocs_nc = {}
display_ylocs_nc = {}
display_zlocs_nc = {}

for i, ch in enumerate(['R1', 'R2', 'R3', 'R4']):
    
    my_df_noncluster = pd.read_hdf(path + fname_noncluster[ch], key='locs')
    my_df_cluster = pd.read_hdf(path + fname_cluster[ch], key='locs')  

    my_df_noncluster.x = my_df_noncluster.x * px_size
    my_df_noncluster.y = my_df_noncluster.y * px_size
    
    my_df_cluster.x = my_df_cluster.x * px_size
    my_df_cluster.y = my_df_cluster.y * px_size
    
    locs_nc_x[ch] = my_df_noncluster.x
    locs_nc_y[ch] = my_df_noncluster.y
    locs_nc_z[ch] = my_df_noncluster.z
    
    locs_c_x[ch] = my_df_cluster.x
    locs_c_y[ch] = my_df_cluster.y
    locs_c_z[ch] = my_df_cluster.z

plt.style.use('dark_background')
fig, ax = plt.subplots()
ax.set_aspect('equal')
    
colors = ['#CC99C9', '#9EC1CF', '#9EE09E', '#FEB144']
display3d = False

if display3d:
    
    #TODO: fix 3D visualization
    
    for i, ch in enumerate(['R1', 'R2', 'R3', 'R4']):
        
        ax = fig.add_subplot(projection='3d')
        
        xmin, xmax = 30000, 34000
        ymin, ymax = 33000, 37000
        roi = (locs_nc_y[ch] > ymin) & (locs_nc_y[ch] < ymax) & (locs_nc_x[ch] > xmin) & (locs_nc_x[ch] < xmax)
        
        display_xlocs_nc[ch] = locs_nc_x[ch][roi]
        # display_xlocs_c[ch] = locs_c_x[ch][roi]

        display_ylocs_nc[ch] = locs_nc_y[ch][roi]
        # display_ylocs_c[ch] = locs_c_y[ch][roi]
        
        display_zlocs_nc[ch] = locs_nc_z[ch][roi]
        # display_zlocs_c[ch] = locs_c_z[ch][roi]

        
        ax.scatter(display_xlocs_nc[ch], display_ylocs_nc[ch], display_zlocs_nc[ch], color=colors[i], edgecolors='white', alpha=0.3)
        # ax.scatter(display_xlocs_c[ch], display_ylocs_c[ch], display_zlocs_c[ch], color=colors[i], edgecolors='white')
        # ax.set_xlim(32000, 33000)
        # ax.set_ylim(35000, 36000)
    
else:

    for i, ch in enumerate(['R1', 'R2', 'R3', 'R4']):
    
        ax.scatter(locs_nc_x[ch], locs_nc_y[ch], color=colors[i], edgecolors='white', alpha=0.3)
        ax.scatter(locs_c_x[ch], locs_c_y[ch], color=colors[i], edgecolors='white')
        ax.set_xlim(30000, 35000)
        ax.set_ylim(35000, 40000)