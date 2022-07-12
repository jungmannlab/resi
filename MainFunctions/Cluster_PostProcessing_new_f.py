#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:15:56 2018

@author: thomasschlichthaerle
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 22:21:32 2018

@author: thomasschlichthaerle
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This Python scripts displays picked regions and let's you slice them in Z

1) Read in 2xhdf5 cluster Files generated from DBSCAN
2) Go through clusters and find nearest cluster in same file (a & b) -> save nearest cluster distance
2a) Go through clusters and measure 1st nearest neighbor, 2nd, 3rd, 4th, 5th etc.


3) Go through clusters and find nearest clusters in the other file (a looks in b ; b looks in a) -> save nearest cluster distance
3a) Go through and find 1st, 2nd, 3rd, 4th, 5th neareast neighbor

4) Calculate a local density map, which uses a threshold distance (how many neighbors at which distance??) to the same protein
4a) Threshold distance
4b) Write a colored map, with each spot with x neighbors is a red dot etc.


5) Calculate a local density map, which uses a threshold distance to another protein of interest!
5a) Threshold distance
5b) Write a colored map, with each spot with x neighbors is a red dot etc.


6) Write a code which analyses colocalization
a) How does local density correlate between two proteins (scatterplot density prot. A, density prot B.) - a rolling region over the whole field of view with diameter x
b) Compare this with global density of random distribution and scatterplot it
c) Find the amount of proteins surrounding (density prot. A versus density of prot B., starting always from the same protein - how many neighbors within radius X)




"HDF5 - members: picked "
STANDARD HDF5 FILE
Frame - 0, x - 1, y - 2, photons-3, sx-4, sy-5, bg-6, lpx-7, lpy-8, ellipticity-9, net_gradient-10, group pick - 11
print(data[1][1])


DBSCAN CLUSTERS FILE:
    ++++++
    Groups - 0, convex_hull - 1, area -2, mean_frame - 3, Center of Mass (com_x) - 4, com_y - 5, std_frame - 6, std_x - 7, std_y - 8, n - 9
    
    +++++

"""


#Read in HDF5-File
from timeit import default_timer as timer
start_all=timer()

import numpy as np
import sys
#import seaborn as sns
#from sklearn.cluster import DBSCAN
import math
from numba import cuda
import os
import h5py
import pandas as pd

import tools
    
'''Functions
'''
    
    
#detects 1st, 2nd... 10th nearest neighbors 
@cuda.jit
def NNDs_kernel(x_com, y_com, higher_neighbors_data, z_com=np.zeros(1)):

    i = cuda.grid(1)
    
    if (i >= len(higher_neighbors_data)):
        return
    
    current_distance = 10000

    for j in range(0, len(x_com)):
        if z_com.size == 1:
            current_distance = math.sqrt((x_com[i]-x_com[j])**2+
                                         (y_com[i]-y_com[j])**2)*130
        else:
            current_distance = math.sqrt((x_com[i]-x_com[j])**2+
                                         (y_com[i]-y_com[j])**2+
                                         (z_com[i]-z_com[j])**2)*130

        if j!= i:
            row_done = False
            for z in range(0,len(higher_neighbors_data[1])):
                
                if z == 0 and not row_done and current_distance < higher_neighbors_data[i][z]:
                        for k in range(len(higher_neighbors_data[1])-1, z,-1):
                            higher_neighbors_data[i][k] =  higher_neighbors_data[i][k-1]
                        higher_neighbors_data[i][z] = current_distance
                        row_done = True
                      
                if z!= 0 and not row_done and current_distance < higher_neighbors_data[i][z] and current_distance > higher_neighbors_data[i][z-1]:
                        for k in range(len(higher_neighbors_data[1])-1, z,-1):
                            higher_neighbors_data[i][k] =  higher_neighbors_data[i][k-1]
                        higher_neighbors_data[i][z] = current_distance
                        row_done = True
  
#detects 1st, 2nd... 10th nearest neighbors between two species
@cuda.jit
def NNDs_ex_kernel(x_com1, y_com1, x_com2, y_com2, higher_neighbors_data, 
                   crossNND_partner_ID, crossNND_partner_x, crossNND_partner_y,
                   z_com1=np.zeros(1), z_com2=np.zeros(1), crossNND_partner_z=np.zeros(1)):

    i = cuda.grid(1)
    
    if (i >= len(higher_neighbors_data)):
        return
    
    current_distance = 10000

    for j in range(0, len(x_com2)):
        if z_com1.size == 1:
            current_distance = math.sqrt((x_com1[i]-x_com2[j])**2+
                                         (y_com1[i]-y_com2[j])**2)*130
        else:
            current_distance = math.sqrt((x_com1[i]-x_com2[j])**2+
                                         (y_com1[i]-y_com2[j])**2+
                                         (z_com1[i]-z_com2[j])**2)*130

        row_done = False
        for z in range(0,len(higher_neighbors_data[1])):
            
            if z == 0 and not row_done and current_distance < higher_neighbors_data[i][z]:
                    for k in range(len(higher_neighbors_data[1])-1, z,-1):
                        higher_neighbors_data[i][k] =  higher_neighbors_data[i][k-1]
                    higher_neighbors_data[i][z] = current_distance
                    crossNND_partner_ID[i] = j
                    crossNND_partner_x[i] = x_com2[j]
                    crossNND_partner_y[i] = y_com2[j]
                    crossNND_partner_z[i] = z_com2[j]
                    row_done = True
                  
            if z!= 0 and not row_done and current_distance < higher_neighbors_data[i][z] and current_distance > higher_neighbors_data[i][z-1]:
                    for k in range(len(higher_neighbors_data[1])-1, z,-1):
                        higher_neighbors_data[i][k] =  higher_neighbors_data[i][k-1]
                    higher_neighbors_data[i][z] = current_distance
                    row_done = True

    
#calculate amount of neighbors within a range of radii  
@cuda.jit
def amountofNeighbors_kernel(x_com, y_com, amountOfNeighbors_data, threshold_radius, z_com=np.zeros(1)):
    i = cuda.grid(1)
    
    if (i >= len(amountOfNeighbors_data)):
        return    

    for j in range(0, len(x_com)):
        if i!=j:
            if z_com.size == 1:
                current_distance = math.sqrt((x_com[i]-x_com[j])**2+
                                             (y_com[i]-y_com[j])**2)*130
            else:
                current_distance = math.sqrt((x_com[i]-x_com[j])**2+
                                             (y_com[i]-y_com[j])**2+
                                             (z_com[i]-z_com[j])**2)*130
            
            if current_distance <= threshold_radius[len(threshold_radius)-1]:
                for k in range(0,len(threshold_radius)):
                    if current_distance <= threshold_radius[k]:
                        amountOfNeighbors_data[i][k] += 1


# the same across to species: how many species B neighbors does have the centered species A protein within various radii
@cuda.jit
def amountofNeighbors_ex_kernel(x_com1, y_com1, x_com2, y_com2, amountOfNeighbors_data, 
                                threshold_radius, z_com1=np.zeros(1), z_com2=np.zeros(1)):
    i = cuda.grid(1)
    
    if (i >= len(amountOfNeighbors_data)):
        return    

    for j in range(0, len(x_com2)):

        if z_com1.size == 1:
            current_distance = math.sqrt((x_com1[i]-x_com2[j])**2+
                                         (y_com1[i]-y_com2[j])**2)*130
        else:
            current_distance = math.sqrt((x_com1[i]-x_com2[j])**2+
                                         (y_com1[i]-y_com2[j])**2+
                                         (z_com1[i]-z_com2[j])**2)*130

        if current_distance <= threshold_radius[len(threshold_radius)-1]:
            for k in range(0,len(threshold_radius)):
                if current_distance <= threshold_radius[k]:
                    amountOfNeighbors_data[i][k] += 1







def postprocessing_cross(protein1, protein2, npz_file1, npz_file2, resi_file1, resi_file2, colocalization_radius):

    default_dist = 1000000


    """load Resi files"""
    f1 = h5py.File(resi_file1, 'r')
    a_group_key = list(f1.keys())[0]
    resi1 = np.array(f1[a_group_key])
    df_resi1 = pd.DataFrame(resi1)


    f2 = h5py.File(resi_file2, 'r')
    a_group_key = list(f2.keys())[0]
    resi2 = np.array(f2[a_group_key])
    df_resi2 = pd.DataFrame(resi2)
    
    
    # Load NPZ file 1 "Talin"
    filename = npz_file1
    f1 = np.load(filename)
    x_com1 = f1['new_com_x_cluster']   
    y_com1 = f1['new_com_y_cluster']

    try:
        z_com1 = f1['new_com_z_cluster'] # in px
        print("3D data detected.")
        flag_3D_1 = True
    except:
        print("2D data detected.")
        flag_3D_1 = False
        
    
    NN_Talin_T = np.zeros(len(x_com1))


    
    # Load NPZ file 1 "Kindlin"
    
    filename2 = npz_file2
    f2 = np.load(filename2)
    
    x_com2 = f2['new_com_x_cluster']
    y_com2 = f2['new_com_y_cluster']

    try:
        z_com2 = f2['new_com_z_cluster']
        print("Second dataset: 3D data detected.")
        flag_3D_2 = True
    except:
        print("Second dataset: 2D data detected.")
        flag_3D_2 = False
    
    if not flag_3D_1 == flag_3D_2:
        sys.exit("Both datasets have to be either 2D or 3D.")
    
    flag_3D = flag_3D_1

    
    NN_Kindlin_K = np.zeros(len(x_com2))
    NN_Talin_Kindlin = np.zeros(len(x_com1))
    NN_Kindlin_Talin = np.zeros(len(x_com2))
    
    coloc_def = 0
    
    coloc_def = int(colocalization_radius)
    
    d_x_com1 = cuda.to_device(x_com1)
    d_y_com1 = cuda.to_device(y_com1)
    d_x_com2 = cuda.to_device(x_com2)
    d_y_com2 = cuda.to_device(y_com2)
    if flag_3D:
        d_z_com1 = cuda.to_device(z_com1)  
        d_z_com2 = cuda.to_device(z_com2) 
    # A dummy variable for numba's 'optional' arguments
    dum_arr = np.zeros(1)       
    
    '''===============================
    #3 check distance to nearest neighbour in other file and save nearest neighbour distance for Talin to Kindlin
    =========================='''
    
    '''Kindlin2 to Talin calculation'''
    
             
    '''    
    #higher_order_nearest_neighbors Kindlin to talin
    '''
    
    #if filename2.isnumeric() == False:
    higher_neighbors_kindlin_to_talin = np.zeros((len(x_com2),10),dtype=float) + default_dist
    crossNND_partner_ID_kindlin_to_talin = np.zeros(len(x_com2),dtype=int)
    crossNND_partner_x_kindlin_to_talin = np.zeros(len(x_com2),dtype=float)
    crossNND_partner_y_kindlin_to_talin = np.zeros(len(x_com2),dtype=float)
    if flag_3D:
        crossNND_partner_z_kindlin_to_talin = np.zeros(len(x_com2),dtype=float)


    #d_higher_neighbors_kindlin_to_talin = cuda.to_device(higher_neighbors_kindlin_to_talin)
    
    
    blockdim_1d = 32
    griddim_1d = len(x_com2)//blockdim_1d + 1
    
    if not flag_3D:
        NNDs_ex_kernel[griddim_1d, blockdim_1d](d_x_com2, d_y_com2, d_x_com1, d_y_com1, 
            higher_neighbors_kindlin_to_talin, crossNND_partner_ID_kindlin_to_talin, 
            crossNND_partner_x_kindlin_to_talin, crossNND_partner_y_kindlin_to_talin, 
            dum_arr, dum_arr, dum_arr)
    
    else:
        NNDs_ex_kernel[griddim_1d, blockdim_1d](d_x_com2, d_y_com2, d_x_com1, d_y_com1, 
            higher_neighbors_kindlin_to_talin, crossNND_partner_ID_kindlin_to_talin, 
            crossNND_partner_x_kindlin_to_talin, crossNND_partner_y_kindlin_to_talin, 
            d_z_com2, d_z_com1, crossNND_partner_z_kindlin_to_talin)
    
    #higher_neighbors_kindlin_to_talin = d_higher_neighbors_kindlin_to_talin.copy_to_host()



    NN_Kindlin_Talin = higher_neighbors_kindlin_to_talin[:,0]
    
    coloc_counter = 0

    for i in range(0, len(NN_Kindlin_Talin)):
        if NN_Kindlin_Talin[i] <= coloc_def:
            coloc_counter = coloc_counter + 1
    
    
    #Labeling_efficiency2_to_1 = np.zeros(2)
    #Labeling_efficiency2_to_1[0] = coloc_counter/len(NN_Kindlin_Talin)*100
    Labeling_efficiency_K2_T1 = coloc_counter/len(NN_Kindlin_Talin)*100

    
        
    
    '''Talin to Kindlin calculation'''
    
    #higher_order_nearest_neighbors Talin to Kindlin

    higher_neighbors_talin_to_kindlin = np.zeros((len(x_com1),10),dtype=float) + default_dist
    crossNND_partner_ID_talin_to_kindlin = np.zeros(len(x_com1),dtype=int)
    crossNND_partner_x_talin_to_kindlin = np.zeros(len(x_com1),dtype=float)
    crossNND_partner_y_talin_to_kindlin = np.zeros(len(x_com1),dtype=float)
    if flag_3D:
        crossNND_partner_z_talin_to_kindlin = np.zeros(len(x_com1),dtype=float)

    
    blockdim_1d = 32
    griddim_1d = len(x_com2)//blockdim_1d + 1
    
    if not flag_3D:
        NNDs_ex_kernel[griddim_1d, blockdim_1d](d_x_com1, d_y_com1, d_x_com2, d_y_com2, 
            higher_neighbors_talin_to_kindlin, crossNND_partner_ID_talin_to_kindlin, 
            crossNND_partner_x_talin_to_kindlin, crossNND_partner_y_talin_to_kindlin,
            dum_arr, dum_arr, dum_arr)

    
    else:
        NNDs_ex_kernel[griddim_1d, blockdim_1d](d_x_com1, d_y_com1, d_x_com2, d_y_com2, 
            higher_neighbors_talin_to_kindlin, crossNND_partner_ID_talin_to_kindlin, 
            crossNND_partner_x_talin_to_kindlin, crossNND_partner_y_talin_to_kindlin,
            d_z_com1, d_z_com2, crossNND_partner_z_talin_to_kindlin)
    


    NN_Talin_Kindlin = higher_neighbors_talin_to_kindlin[:,0]
    coloc_counter = 0

    for i in range(0, len(NN_Talin_Kindlin)):
        if NN_Talin_Kindlin[i] <= coloc_def:
            coloc_counter = coloc_counter + 1
        
        
    #Labeling_efficiency1_to_2 = np.zeros(2)
    #Labeling_efficiency1_to_2[0] = coloc_counter/len(NN_Talin_Kindlin)*100    
    Labeling_efficiency_T1_K2 = coloc_counter/len(NN_Talin_Kindlin)*100    
        
    


    path =  os.path.split(filename)[0] + "/AdditionalOutputs/"
    fname = path + os.path.split(filename)[1]
    fname2 = path + os.path.split(filename2)[1]

    #np.savetxt('%s_coloc_Percentages_%s_to_%s.txt' %(fname2, protein2, protein1), Labeling_efficiency2_to_1, delimiter= ',')
    #np.savetxt('%s_coloc_Percentages_%s_to_%s.txt' %(fname, protein1, protein2), Labeling_efficiency1_to_2, delimiter= ',')
    with open('%s_coloc_Percentage_%s_to_%s.txt' %(fname2, protein2, protein1), 'w') as f:
        f.write(str(Labeling_efficiency_K2_T1))
    with open('%s_coloc_Percentage_%s_to_%s.txt' %(fname, protein1, protein2), 'w') as f:
        f.write(str(Labeling_efficiency_T1_K2))

    NNA = np.savetxt('%s_Neighbor_distance_%s_to_%s.csv' %(fname, protein1, protein2), NN_Talin_Kindlin)
    NNA = np.savetxt('%s_Neighbor_distance_%s_to_%s.csv' %(fname2, protein2, protein1), NN_Kindlin_Talin)
    
    NNA = np.savetxt('%s_higher_Neighbors_%s_to_%s.csv' %(fname2, protein2, protein1), higher_neighbors_kindlin_to_talin, delimiter= ',')
    NNA = np.savetxt('%s_higher_Neighbors_%s_to_%s.csv' %(fname, protein1, protein2), higher_neighbors_talin_to_kindlin, delimiter= ',')
 


    df_resi1['crossNND_ID'] = crossNND_partner_ID_talin_to_kindlin
    df_resi1['crossNND_x'] = crossNND_partner_x_talin_to_kindlin
    df_resi1['crossNND_y'] = crossNND_partner_y_talin_to_kindlin
    if flag_3D:
        df_resi1['crossNND_z'] = crossNND_partner_z_talin_to_kindlin # in px
    df_resi1['crossNND'] = NN_Talin_Kindlin
    if flag_3D:
        df_resi1['crossNND_dxy'] = np.sqrt((df_resi1['x']-df_resi1['crossNND_x'])**2+
                                             (df_resi1['y']-df_resi1['crossNND_y'])**2)*130
        df_resi1['crossNND_dz'] = (df_resi1['z']/130-df_resi1['crossNND_z'])*130
    df_resi1['orientation'] = tools.angle(df_resi1['x'], df_resi1['y'], df_resi1['crossNND_x'], df_resi1['crossNND_y'])


    path = os.path.split(filename)[0] + "/"
    filename_old = os.path.split(resi_file1)[1]
    filename_new = '%s_info.hdf5' % (filename_old[:-5])
    tools.picasso_hdf5(df_resi1, filename_new, filename_old, path)
    
    
    """save Kindlin"""

    df_resi2['crossNND_ID'] = crossNND_partner_ID_kindlin_to_talin
    df_resi2['crossNND_x'] = crossNND_partner_x_kindlin_to_talin
    df_resi2['crossNND_y'] = crossNND_partner_y_kindlin_to_talin
    if flag_3D:
        df_resi2['crossNND_z'] = crossNND_partner_z_kindlin_to_talin # in px
    df_resi2['crossNND'] = NN_Kindlin_Talin
    if flag_3D:
        df_resi2['crossNND_dxy'] = np.sqrt((df_resi2['x']-df_resi2['crossNND_x'])**2+
                                             (df_resi2['y']-df_resi2['crossNND_y'])**2)*130
        df_resi2['crossNND_dz'] = (df_resi2['z']/130-df_resi2['crossNND_z'])*130
    df_resi2['orientation'] = tools.angle(df_resi2['x'], df_resi2['y'], df_resi2['crossNND_x'], df_resi2['crossNND_y'])

    

    path = os.path.split(filename)[0] + "/"
    filename_old = os.path.split(resi_file2)[1]
    filename_new = '%s_info.hdf5' % (filename_old[:-5])
    tools.picasso_hdf5(df_resi2, filename_new, filename_old, path)






def postprocessing(protein, npz_file1, colocalization_radius):

    
    filename = npz_file1
    f1 = np.load(filename)
    
    x_com1 = f1['new_com_x_cluster']   
    y_com1 = f1['new_com_y_cluster']
    
    try:
        z_com1 = f1['new_com_z_cluster']
        print("3D data detected.")
        flag_3D = True
    except:
        print("2D data detected.")
        flag_3D = False
        
    
    NN_Talin_T = np.zeros(len(x_com1))
    

    coloc_def = 0
    coloc_def = int(colocalization_radius)
    
    # A dummy variable for numba's 'optional' arguments
    dum_arr = np.zeros(1)
    

    '''Talin calculation'''
    default_dist = 1000000
    higher_neighbors_Talin = np.zeros((len(x_com1),10),dtype=float) + default_dist #this gives you the 10 nearest neighbors
    
    
    d_x_com1 = cuda.to_device(x_com1)
    d_y_com1 = cuda.to_device(y_com1)
    #d_higher_neighbors_Talin = cuda.to_device(higher_neighbors_Talin)
    if flag_3D:
        d_z_com1 = cuda.to_device(z_com1)
    
    blockdim_1d = 32
    griddim_1d = len(x_com1)//blockdim_1d + 1
    
    if not flag_3D:
        NNDs_kernel[griddim_1d, blockdim_1d](d_x_com1, d_y_com1, higher_neighbors_Talin, dum_arr)
    else:
        NNDs_kernel[griddim_1d, blockdim_1d](d_x_com1, d_y_com1, higher_neighbors_Talin, d_z_com1)
        

    
    
    NN_Talin_T = higher_neighbors_Talin[:,0]
    
    #calculate percentage of talins whose NND is smaller than coloc_def
    coloc_counter = 0
    
    for i in range(0, len(NN_Talin_T)):
        if NN_Talin_T[i] <= coloc_def: #here is the distance
            coloc_counter = coloc_counter + 1
            
            

    #Labeling_efficiency2_to_1 = np.zeros(2)
    #Labeling_efficiency2_to_1[0] = coloc_counter/len(NN_Talin_T)*100 # percentage of talins that have another talin as a neighbor within the coloc_def distance
    Labeling_efficiency_T1 = coloc_counter/len(NN_Talin_T)*100 # percentage of talins that have another talin as a neighbor within the coloc_def distance
                 

        
    try:
        os.mkdir(os.path.split(filename)[0] + '/AdditionalOutputs')
    
    except OSError:
        print ("AdditionalOutputs folder already exists")
    path =  os.path.split(filename)[0] + "/AdditionalOutputs/"
    fname = path + os.path.split(filename)[1]
    #np.savetxt('%s_coloc_Percentage_%s.txt' %(fname, protein), Labeling_efficiency2_to_1, delimiter= ',')
    with open('%s_coloc_Percentage_%s.txt' %(fname, protein), 'w') as f:
        f.write(str(Labeling_efficiency_T1))
    
    NNA = np.savetxt('%s_Neighbor_distance_%s.csv' %(fname, protein), NN_Talin_T)
    NNA = np.savetxt('%s_higher_Neighbors_%s.csv' %(fname, protein), higher_neighbors_Talin, delimiter= ',')
