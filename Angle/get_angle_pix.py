# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 17:13:27 2022

@author: reinhardt
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:29:36 2022

@author: reinhardt
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from os.path import dirname as up

cwd = os.getcwd()
wdir = up(cwd)
os.chdir(wdir)
import tools


"""
The input data was created via the following procedure:
1. Merge all Resi info.hdf5 files for the R1 and the R3 channel respectively
2. Load the R1 (or R3 - but one of them is sufficient) file containing the 
   originally picked origamis in their raw, unclustered form ("R1..._apicked.hdf5)
   into Picasso Average. Create an average structure of the R1 origamis. 
3. Apply the transformation determined by Picasso Average in step 2 to the 
   merged resi info.hdf5 files from step1 (It is important that the same filees
   from step2 serve as the template for applying the transforamtion to both the 
   R1 and R3 files - no matter if the template was created with R1 or R3 data. 
   This is important such that the relative orientation of the R1 and R3 
   origamis in the merged files is maintained.)
--- The previous steps are normally already done when you want to create an 
    overlay image of the RESI cluster centers from all origamis to measure 
    the distance between R1/R3 sites in this summed image of all oris.----
--- We will now start from these aligned R1 and R3 merged Resi-info files. 
    While we actually do not need the previous averaging step as we now anyways
    will fit each origami's cluster centers to the model coordinates, it has 
    a practical reason why we don't start from the un-averaged origami file(s):
    It is important to identify which sites from the model are missing in one
    of the origamis to properly perform the fit. 
    The LQ fitting code requires the same number of coordinates in the model 
    as in the data to be fitted to the model. Thus, I will identify which sites
    are missing in the origami and then delete these in the model that will
    be used for this origami.
4. In order to identify which localization in the origami belong to which position
   in the model, the 11 sites in the averaged Resi Info files will be picked, 
   such that the group information corresponds to the position of the localization.
   Make sure to pick the sites in a very specific order (see numbers next to the 
   sites in the model). Otherwise the localizations can't be assigned to their
   corresponding position in the model. 
"""

"""
What the script will do:
1. Fit the R3 cluster centers of each origami to the model. (This is done with
   the R3 channel as all sites in this channel are 3' extended and thus the coordinates
   of the design are better known. While we do not exactly know how to handle
   the R1 5' extensions in a model.)
"""

"""
Note: All plots are mirrored in y direction compared to the Picasso display!
"""

'''
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(121, title='Model with position numbers')
ax.scatter(R3_model_x, R3_model_y, color='r', s = 5)
ax.set_aspect('equal')
ax.legend()
for i, label in enumerate(group_number):
    ax.annotate(str(round(label,2)), (R3_model_x[i], R3_model_y[i]))
'''


# Data import
path = r'W:\users\reinhardt\z_raw\Resi\211123_dbl-ext-20nm-6sites-R4\workflow_081221\fit_to_template\fit_every_ori_to_model'

fov_size = 512 # size of the field of view in pixel
pixel_size = 130
R1_filename = os.path.join(path,'R1_apicked_ori-all_aligned_resi_4_50_info_merge_avg-appl_withOriID_apicked.hdf5')
R3_filename = os.path.join(path,'R3_apicked_ori-all_aligned_resi_4_50_info_merge_avg-appl_withOriID_apicked.hdf5')

R3_model = pd.DataFrame()
R3_model['x_m'] = np.array([25.000000, 45.000000, 75.000000, 0.000000, 25.000000, 45.000000, 65.000000, 75.000000, 0.000000, 25.000000, 45.000000, 75.000000])/pixel_size + fov_size/2
R3_model['y_m'] = -np.array([10.000000, 10.000000, 10.000000, 22.500000, 30.000000, 30.000000, 30.000000, 30.000000, 37.500000, 50.000000, 50.000000, 55.000000])/pixel_size - fov_size/2
R3_model['position_number_m'] = np.array([8,11,5,1,7,10,3,4,0,6,9,2])
print(R3_model)

plots_path = path + '/plots'
try:
    os.mkdir(plots_path)

except OSError:
    print ("Plots folder already exists")


R1_locs = pd.read_hdf(R1_filename, key = 'locs')
R3_locs = pd.read_hdf(R3_filename, key = 'locs')
#print(R1_locs.keys())

print(R3_locs.head())

R1_grouped = R1_locs.groupby('Origami_ID')
R3_grouped = R3_locs.groupby('Origami_ID')
print(R3_grouped.get_group(0).head())


R3_locs_transfo = R3_locs.copy()
R1_locs_transfo = R1_locs.copy()
list_R3_transfo = []
list_R1_transfo = []
if 'crossNND_x' in R3_locs.keys():
    list_R3_transfo_cross = []
    list_R1_transfo_cross = []

result_R1 = []
result_R3 = []
for ori_number, R3_group in R3_grouped:
    R1_group = R1_grouped.get_group(ori_number)
    R1_group_new = R1_group.copy()
    R3_group_new = R3_group.copy()
    
    # Merge the columns of the R3_model and the R3_group dataframe. 
    # The rows are matched by the position of the localization in the origami 
    # meaning the group column in the real data.
    R3_joined = R3_model.merge(R3_group, how = 'inner', left_on = 'position_number_m', right_on = 'group')
    #print(R3_joined)
    
    # Perform LQ alignment to the model and get the rotation R and translation t parameters 
    # R model + t = B
    R3_model_xy = np.transpose(np.array(R3_joined[['x_m','y_m']]))
    R3_data_xy = np.transpose(np.array(R3_joined[['x','y']]))
    R,t = tools.rigid_transform_2D(R3_model_xy, R3_data_xy)
    
    R_inv = np.linalg.inv(R)
    
    # Align the R3 channel to the model
    R3_group_T = np.transpose(np.array(R3_group[['x','y']]))

    R3_group_transfo = np.transpose(R_inv @ (R3_group_T - t))
    
    #list_R3_transfo.extend(list(R3_group_transfo))

    if 'crossNND_x' in R3_locs.keys():
        R3_data_xy_cross = np.transpose(np.array(R3_group[['crossNND_x','crossNND_y']])) # so it's actually R1 data
        R3_group_transfo_cross = np.transpose(R_inv @ (R3_data_xy_cross - t))
        #list_R3_transfo_cross.extend(list(np.transpose(R3_group_transfo_cross)))
    
    # Transform the R1 channel in the same way as the R3 channel 
    R1_group_T = np.transpose(np.array(R1_group[['x','y']]))

    R1_group_transfo = np.transpose(R_inv @ (R1_group_T - t))
    
    #list_R1_transfo.extend(list(R1_group_transfo))

    if 'crossNND_x' in R1_locs.keys():
        R1_data_xy_cross = np.transpose(np.array(R1_group[['crossNND_x','crossNND_y']])) # so it's actually R3 data
        R1_group_transfo_cross = np.transpose(R_inv @ (R1_data_xy_cross - t))
        #list_R1_transfo_cross.extend(list(np.transpose(R1_group_transfo_cross)))
            
    
    fig = plt.figure(figsize=(30,10))
    plt.suptitle('Origami ' + str(ori_number), fontsize = 20)

    ax = fig.add_subplot(131, title='Model with position numbers')
    ax.scatter(R3_model['x_m'], R3_model['y_m'], color='g', s = 5)
    ax.set_aspect('equal')
    ax.legend()
    for i, label in enumerate(R3_model['position_number_m']):
        ax.annotate(str(round(label,2)), (R3_model['x_m'][i], R3_model['y_m'][i]))
    
    ax2 = fig.add_subplot(132, title='Origami cluster centers and group number')
    ax2.scatter(R3_group['x'], R3_group['y'], color='r', s = 5)
    ax2.set_aspect('equal')
    #print(R3_grouped.get_group(0)['x'])
    # print the group number "the site to which it belongs" next to the coordinate to check if it matches the 
    # numbers in the model
    for i, label in enumerate(R3_group['group']):
        ax2.annotate(str(round(label,2)), (np.array(R3_group['x'])[i], np.array(R3_group['y'])[i]))
    
    # print(R3_group_transfo)
    ax3 = fig.add_subplot(133, title='Model and aligned origami centers')
    ax3.scatter(R3_model['x_m'], R3_model['y_m'], color='g', s = 5)
    ax3.scatter(R3_group_transfo[:,0], R3_group_transfo[:,1], color='b', s = 5)
    ax3.set_aspect('equal')
    plt.savefig(os.path.join(plots_path, 'Origami' + str(ori_number) + '_R3_fit_to_model.png'), transparent=False, bbox_inches='tight')
    plt.savefig(os.path.join(plots_path, 'Origami' + str(ori_number) + '_R3_fit_to_model.pdf'), transparent=False, bbox_inches='tight')
    
    
    fig = plt.figure()
    plt.suptitle('Origami ' + str(ori_number), fontsize = 20)

    ax = fig.add_subplot(111, title='Model and aligned R1 and R3 channels')
    ax.scatter(R3_model['x_m'], R3_model['y_m'], color='g', s = 5, label = 'model')
    ax.scatter(R1_group_transfo[:,0], R1_group_transfo[:,1], color='r', s = 5, label = 'R1')
    ax.scatter(R3_group_transfo[:,0], R3_group_transfo[:,1], color='b', s = 5, label = 'R3')
    ax.set_aspect('equal')
    ax.legend()
    plt.savefig(os.path.join(plots_path, 'Origami' + str(ori_number) + '_R1_R3.png'), transparent=False, bbox_inches='tight')
    plt.savefig(os.path.join(plots_path, 'Origami' + str(ori_number) + '_R1_R3.pdf'), transparent=False, bbox_inches='tight')
    
    
    R1_group_new['x_model'] = R1_group_transfo[:,0]
    R1_group_new['y_model'] = R1_group_transfo[:,1]
    R1_group_new['crossNND_x_model'] = R1_group_transfo_cross[:,0]
    R1_group_new['crossNND_y_model'] = R1_group_transfo_cross[:,1]
    
    R3_group_new['x_model'] = R3_group_transfo[:,0]
    R3_group_new['y_model'] = R3_group_transfo[:,1]
    R3_group_new['crossNND_x_model'] = R3_group_transfo_cross[:,0]
    R3_group_new['crossNND_y_model'] = R3_group_transfo_cross[:,1]
    
    
    result_R1.append(R1_group_new)
    result_R3.append(R3_group_new)
    
    
result_R3 = pd.concat(result_R3)
print(result_R3.head())  
result_R3 = result_R3.sort_index()
print(result_R3.head())    


result_R1 = pd.concat(result_R1)
print(result_R1.head())  
result_R1 = result_R1.sort_index()
print(result_R1.head())   


if 'crossNND_x' in R3_locs.keys():
    result_R3['orientation_model'] = tools.angle(result_R3['x_model'], result_R3['y_model'], result_R3['crossNND_x_model'], result_R3['crossNND_y_model'])

if 'crossNND_x' in R1_locs.keys():
    result_R1['orientation_model'] = tools.angle(result_R1['x_model'], result_R1['y_model'], result_R1['crossNND_x_model'], result_R1['crossNND_y_model'])


R1_name_new = os.path.split(R1_filename)[1]
R1_name_new = R1_name_new[:-5] + "_model.hdf5"
tools.picasso_hdf5(result_R1, R1_name_new, os.path.split(R1_filename)[1], path + "/")

R3_name_new = os.path.split(R3_filename)[1]
R3_name_new = R3_name_new[:-5] + "_model.hdf5"
tools.picasso_hdf5(result_R3, R3_name_new, os.path.split(R3_filename)[1], path + "/")

