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

import tools
    
'''Functions
'''
    
    
    
@cuda.jit
def NNDs_kernel(x_com, y_com, higher_neighbors_data):

    i = cuda.grid(1)
    
    if (i >= len(higher_neighbors_data)):
        return
    
    current_distance = 10000

    for j in range(0, len(x_com)):
        current_distance = math.sqrt((x_com[i]-x_com[j])**2+(y_com[i]-y_com[j])**2)*130
        
        if j!= i:
            for z in range(0,len(higher_neighbors_data[1])):
                
                if z == 0:
                    if higher_neighbors_data[i][z] == 0:
                        higher_neighbors_data[i][z] = current_distance
            
                    if current_distance < higher_neighbors_data[i][z]:
                        higher_neighbors_data[i][z] = current_distance
                      
                if z!= 0:
                    if higher_neighbors_data[i][z] == 0:
                        higher_neighbors_data[i][z] = current_distance
                     
                    if current_distance < higher_neighbors_data[i][z] and current_distance > higher_neighbors_data[i][z-1]:
                        higher_neighbors_data[i][z] = current_distance
  
  
@cuda.jit
def NNDs_ex_kernel(x_com1, y_com1, x_com2, y_com2, higher_neighbors_data):

    i = cuda.grid(1)
    
    if (i >= len(higher_neighbors_data)):
        return
    
    current_distance = 10000

    for j in range(0, len(x_com2)):
        current_distance = math.sqrt((x_com1[i]-x_com2[j])**2+(y_com1[i]-y_com2[j])**2)*130
        
        for z in range(0,len(higher_neighbors_data[1])):
            
            if z == 0:
                if higher_neighbors_data[i][z] == 0:
                    higher_neighbors_data[i][z] = current_distance
        
                if current_distance < higher_neighbors_data[i][z]:
                    higher_neighbors_data[i][z] = current_distance
                  
            if z!= 0:
                if higher_neighbors_data[i][z] == 0:
                    higher_neighbors_data[i][z] = current_distance
                 
                if current_distance < higher_neighbors_data[i][z] and current_distance > higher_neighbors_data[i][z-1]:
                    higher_neighbors_data[i][z] = current_distance
  
  
    
@cuda.jit
def amountofNeighbors_kernel(x_com, y_com, amountOfNeighbors_data, threshold_radius):
    i = cuda.grid(1)
    
    if (i >= len(amountOfNeighbors_data)):
        return    

    for j in range(0, len(x_com)):
        if i!=j:
            current_distance = math.sqrt((x_com[i]-x_com[j])**2+(y_com[i]- y_com[j])**2)*130
            if current_distance <= threshold_radius[len(threshold_radius)-1]:
                for k in range(0,len(threshold_radius)):
                    if current_distance <= threshold_radius[k]:
                        amountOfNeighbors_data[i][k] += 1

@cuda.jit
def amountofNeighbors_ex_kernel(x_com1, y_com1, x_com2, y_com2, amountOfNeighbors_data, threshold_radius):
    i = cuda.grid(1)
    
    if (i >= len(amountOfNeighbors_data)):
        return    

    for j in range(0, len(x_com1)):

        current_distance = math.sqrt((x_com1[i]-x_com2[j])**2+(y_com1[i]-y_com2[j])**2)*130
        if current_distance <= threshold_radius[len(threshold_radius)-1]:
            for k in range(0,len(threshold_radius)):
                if current_distance <= threshold_radius[k]:
                    amountOfNeighbors_data[i][k] += 1






import h5py

def postprocessing_cross(protein1, protein2, npz_file1, npz_file2, resi_file1, resi_file2, colocalization_radius):

    import pandas as pd

    """load Resi files"""
    f1 = h5py.File(resi_file1, 'r')
    a_group_key = list(f1.keys())[0]
    resi1 = np.array(f1[a_group_key])
    df_resi1 = pd.DataFrame(resi1)


    f2 = h5py.File(resi_file2, 'r')
    a_group_key = list(f2.keys())[0]
    resi2 = np.array(f2[a_group_key])
    df_resi2 = pd.DataFrame(resi2)
    
    
    
    
    '''Read in Files'''
    #File 1 - Talin -> truepos
    
    
    filename = npz_file1
    
    
    f1 = np.load(filename)
    
    #data = np.zeros((len(f1['new_com_x_cluster']),6))
    
    '''
    s=timer()
    for i in range(0,len(f1['new_com_x_cluster'])):
        data[i][4] = f1['new_com_x_cluster'][i]
        data[i][5] = f1['new_com_y_cluster'][i]
    e = timer()
    print("old import:", e-s) 
    '''
    
    
    #data_t = np.zeros((len(f1['new_com_x_cluster']),6))
    s=timer()
    x_com1 = f1['new_com_x_cluster']   
    y_com1 = f1['new_com_y_cluster']
    e = timer()
    print("new import:", e-s)
    #len(x_com1)
    #len(x_com2)
    
    '''
    for i in range(0,len(f1['new_com_x_cluster'])):
        if (data[i][4] != x_com[i]):
            print(i, data[i][4], x_com[i])
    '''
    #Get data
    
    
    
    NN_Talin_T = np.zeros(len(x_com1))
    
    # File 2 - Kindlin
    
    
    filename2 = npz_file2
    

    f2 = np.load(filename2)

    #data2 = np.zeros((len(f2['new_com_x_cluster']),6))

    
    x_com2 = f2['new_com_x_cluster']
    y_com2 = f2['new_com_y_cluster']
    
    NN_Kindlin_K = np.zeros(len(x_com2))
    NN_Talin_Kindlin = np.zeros(len(x_com1))
    NN_Kindlin_Talin = np.zeros(len(x_com2))
    
    coloc_def = 0
    
    coloc_def = int(colocalization_radius)
    
    #if filename2.isnumeric() == True:
    #    coloc_def = int(sys.argv[2])
    
    
    d_x_com1 = cuda.to_device(x_com1)
    d_y_com1 = cuda.to_device(y_com1)
    d_x_com2 = cuda.to_device(x_com2)
    d_y_com2 = cuda.to_device(y_com2)
        
    
    '''===============================
    #3 check distance to nearest neighbour in other file and save nearest neighbour distance for Talin to Kindlin
    =========================='''
    
    '''Kindlin2 to Talin calculation'''
    
             
    '''    
    #higher_order_nearest_neighbors Kindlin to talin
    '''
    
    #if filename2.isnumeric() == False:
    higher_neighbors_kindlin_to_talin = np.zeros((len(x_com2),10),dtype=float) 
    


    #d_higher_neighbors_kindlin_to_talin = cuda.to_device(higher_neighbors_kindlin_to_talin)
    
    
    blockdim_1d = 32
    griddim_1d = len(x_com2)//blockdim_1d + 1
    
    NNDs_ex_kernel[griddim_1d, blockdim_1d](d_x_com2, d_y_com2, d_x_com1, d_y_com1, higher_neighbors_kindlin_to_talin)
    
    
    #higher_neighbors_kindlin_to_talin = d_higher_neighbors_kindlin_to_talin.copy_to_host()



    NN_Kindlin_Talin = higher_neighbors_kindlin_to_talin[:,0]
    
    coloc_counter = 0

    for i in range(0, len(NN_Kindlin_Talin)):
        if NN_Kindlin_Talin[i] <= coloc_def:
            coloc_counter = coloc_counter + 1
    
    Labeling_efficiency2_to_1 = np.zeros(2)
    Labeling_efficiency2_to_1[0] = coloc_counter/len(NN_Kindlin_Talin)*100
    
    
        
    
    '''Talin to Kindlin calculation'''
    
    #higher_order_nearest_neighbors Talin to Kindlin
    #if filename2.isnumeric() == False:
        
    higher_neighbors_talin_to_kindlin = np.zeros((len(x_com1),10),dtype=float) 
    


    #d_higher_neighbors_talin_to_kindlin = cuda.to_device(higher_neighbors_talin_to_kindlin)
    
    
    blockdim_1d = 32
    griddim_1d = len(x_com1)//blockdim_1d + 1
    
    NNDs_ex_kernel[griddim_1d, blockdim_1d](d_x_com1, d_y_com1, d_x_com2, d_y_com2, higher_neighbors_talin_to_kindlin)
    
    
    #higher_neighbors_talin_to_kindlin = d_higher_neighbors_talin_to_kindlin.copy_to_host()




    NN_Talin_Kindlin = higher_neighbors_talin_to_kindlin[:,0]
    coloc_counter = 0

    for i in range(0, len(NN_Talin_Kindlin)):
        if NN_Talin_Kindlin[i] <= coloc_def:
            coloc_counter = coloc_counter + 1
        
        
#Labeling_efficiency2_to_1 = 0
    Labeling_efficiency1_to_2 = np.zeros(2)
    Labeling_efficiency1_to_2[0] = coloc_counter/len(NN_Talin_Kindlin)*100    
        
        
    
    
    """=================================
    4) Calculate a local density map, which uses a threshold distance (how many neighbors at which distance??)
    4a) Threshold distance
    4b) Write a colored map, with each spot with x neighbors is a red dot etc.
    ================================="""
    
    threshold_radius = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500, 1000, 5000, 20000]) #threshold distance 30 nm in pixel, pixelsize = 130nm
                    
    #check for each cluster how many neighbors it has within this distance
    # Let's start with Talin
    #amountOfNeighbors_data1 = np.zeros(len(data1_x))
    #amountOfNeighbors_data1_old = np.zeros((len(x_com1),len(threshold_radius)),dtype=int)
    
    
    #for i in range(0, len(data1_x)):
    #    amountOfNeighbors_data1.append([0])
    #    for j in range(0, len(threshold_radius)):
    #        amountOfNeighbors_data1[i].append(0)
    
    '''                 
    s=timer()                
    for i in range(0, len(x_com1)):
        for j in range(0, len(x_com1)):
            if i!=j:
                current_distance = math.sqrt((x_com1[i]-x_com1[j])**2+(y_com1[i]- y_com1[j])**2)*130
                if current_distance <= threshold_radius[len(threshold_radius)-1]:
                    for k in range(0,len(threshold_radius)):
                        if current_distance <= threshold_radius[k]:
                            amountOfNeighbors_data1_old[i][k] = amountOfNeighbors_data1_old[i][k] + 1
                    
    e=timer()
    print("amountOfNeighbors old:", e-s)
    '''
    """
    s=timer()                
    amountOfNeighbors_data1 = np.zeros((len(x_com1),len(threshold_radius)),dtype=int)
    
        
        
    blockdim_1d = 32
    griddim_1d = len(amountOfNeighbors_data1)//blockdim_1d + 1
    
    amountofNeighbors_kernel[griddim_1d, blockdim_1d](d_x_com1, d_y_com1, amountOfNeighbors_data1, threshold_radius)
    
        
    
    e=timer()
    print("amountOfNeighbors new:", e-s)
    """
    
    '''
    for i in range(len(amountOfNeighbors_data1_old)):
        for k in range(len(threshold_radius)):
            if amountOfNeighbors_data1_old[i][k] != amountOfNeighbors_data1[i][k]:
                print("wrong", i, amountOfNeighbors_data1_old[i][k], amountOfNeighbors_data1[i][k])
            
    '''        
            
    """
    #if filename2.isnumeric() == False:
    
    amountOfNeighbors_data2 = np.zeros((len(x_com2),len(threshold_radius)),dtype=int)
    
    blockdim_1d = 32
    griddim_1d = len(amountOfNeighbors_data2)//blockdim_1d + 1
    
    amountofNeighbors_kernel[griddim_1d, blockdim_1d](d_x_com2, d_y_com2, amountOfNeighbors_data2, threshold_radius)

             
                


    amountOfNeighbors_data1_to_data2 = np.zeros((len(x_com1),len(threshold_radius)),dtype=int)
    blockdim_1d = 32
    griddim_1d = len(amountOfNeighbors_data1_to_data2)//blockdim_1d + 1
    
    amountofNeighbors_ex_kernel[griddim_1d, blockdim_1d](d_x_com1, d_y_com1, d_x_com2, d_y_com2, amountOfNeighbors_data1_to_data2, threshold_radius)
             
                



    amountOfNeighbors_data2_to_data1 = np.zeros((len(x_com2),len(threshold_radius)),dtype=int)
    blockdim_1d = 32
    griddim_1d = len(amountOfNeighbors_data2_to_data1)//blockdim_1d + 1
    
    amountofNeighbors_ex_kernel[griddim_1d, blockdim_1d](d_x_com2, d_y_com2, d_x_com1, d_y_com1, amountOfNeighbors_data2_to_data1, threshold_radius)
                
    """
    
    
    
    s=timer()    

    path =  os.path.split(filename)[0] + "/AdditionalOutputs/"
    fname = path + os.path.split(filename)[1]
    fname2 = path + os.path.split(filename2)[1]

    np.savetxt('%s_coloc_Percentages_%s_to_%s.txt' %(fname2, protein2, protein1), Labeling_efficiency2_to_1, delimiter= ',')
    np.savetxt('%s_coloc_Percentages_%s_to_%s.txt' %(fname, protein1, protein2), Labeling_efficiency1_to_2, delimiter= ',')
    
    #if filename2.isnumeric() == False:
    NNA = np.savetxt('%s_Neighbor_distance_%s_to_%s.csv' %(fname, protein1, protein2), NN_Talin_Kindlin)
    NNA = np.savetxt('%s_Neighbor_distance_%s_to_%s.csv' %(fname2, protein2, protein1), NN_Kindlin_Talin)
    
    NNA = np.savetxt('%s_higher_Neighbors_%s_to_%s.csv' %(fname2, protein2, protein1), higher_neighbors_kindlin_to_talin, delimiter= ',')
    NNA = np.savetxt('%s_higher_Neighbors_%s_to_%s.csv' %(fname, protein1, protein2), higher_neighbors_talin_to_kindlin, delimiter= ',')
    
    #NNA = np.savetxt('%s_amount_Of_Neighbors_%s_to_%s.csv' %(filename, protein1, protein2), amountOfNeighbors_data1_to_data2, delimiter= ',')
    #NNA = np.savetxt('%s_amount_Of_Neighbors_%s_to_%s.csv' %(filename2, protein2, protein1), amountOfNeighbors_data2_to_data1, delimiter= ',')
    
    
    df_resi1['crossNND'] = NN_Talin_Kindlin

    path = os.path.split(filename)[0] + "/"
    filename_old = os.path.split(resi_file1)[1]
    filename_new = '%s_info.hdf5' % (filename_old[:-5])
    tools.picasso_hdf5(df_resi1, filename_new, filename_old, path)
    
    
    """save Kindlin"""

    df_resi2['crossNND'] = NN_Kindlin_Talin

    path = os.path.split(filename)[0] + "/"
    filename_old = os.path.split(resi_file2)[1]
    filename_new = '%s_info.hdf5' % (filename_old[:-5])
    tools.picasso_hdf5(df_resi2, filename_new, filename_old, path)






def postprocessing(protein, npz_file1, colocalization_radius):

    '''Read in Files'''
    #File 1 - Talin -> truepos
    
    print(npz_file1)
    filename = npz_file1
    
    
    f1 = np.load(filename)

    s=timer()
    x_com1 = f1['new_com_x_cluster']   
    y_com1 = f1['new_com_y_cluster']
    e = timer()
    print("new import:", e-s)
    
    
    NN_Talin_T = np.zeros(len(x_com1))
    
    # File 2 - Kindlin
    
    
    
    coloc_def = 0
    
    coloc_def = int(colocalization_radius)
    

    
    '''====================================='''
    
    '''#2 = Go through clusters and find nearest neighbour in same file ; save distance to nearest neighbour in new file
    ==========================='''
    
    '''Talin calculation'''
    
                 
                    
    '''====================================='''
    
    '''#2a = find nearest neighbor in same file (1st, 2nd, 3rd, 4th, 5th)
    ==========================='''              
    
    higher_neighbors_Talin = np.zeros((len(x_com1),10),dtype=float) #this gives you the 10 nearest neighbors
    
    
    
    d_x_com1 = cuda.to_device(x_com1)
    d_y_com1 = cuda.to_device(y_com1)
    #d_higher_neighbors_Talin = cuda.to_device(higher_neighbors_Talin)
    
    
    
    
    blockdim_1d = 32
    griddim_1d = len(x_com1)//blockdim_1d + 1
    
    
    NNDs_kernel[griddim_1d, blockdim_1d](d_x_com1, d_y_com1, higher_neighbors_Talin)
    
    
    #higher_neighbors_Talin = d_higher_neighbors_Talin.copy_to_host()
    
    
    
    
    
    
    NN_Talin_T = higher_neighbors_Talin[:,0]
    
    coloc_counter = 0
    
    for i in range(0, len(NN_Talin_T)):
        if NN_Talin_T[i] <= coloc_def: #here is the distance
            coloc_counter = coloc_counter + 1
            
            

    Labeling_efficiency2_to_1 = np.zeros(2)
    Labeling_efficiency2_to_1[0] = coloc_counter/len(NN_Talin_T)*100 # percentage of talins that have another talin as a neighbor within the coloc_def distance
                 

        
    
    
    """=================================
    4) Calculate a local density map, which uses a threshold distance (how many neighbors at which distance??)
    4a) Threshold distance
    4b) Write a colored map, with each spot with x neighbors is a red dot etc.
    ================================="""
    
    threshold_radius = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500, 1000, 5000, 20000]) #threshold distance 30 nm in pixel, pixelsize = 130nm
                    
    #check for each cluster how many neighbors it has within this distance
    # Let's start with Talin
    #amountOfNeighbors_data1 = np.zeros(len(data1_x))
    
    """amountOfNeighbors_data1_old = np.zeros((len(x_com1),len(threshold_radius)),dtype=int)
    """
    
    #for i in range(0, len(data1_x)):
    #    amountOfNeighbors_data1.append([0])
    #    for j in range(0, len(threshold_radius)):
    #        amountOfNeighbors_data1[i].append(0)
    
    '''                 
    s=timer()                
    for i in range(0, len(x_com1)):
        for j in range(0, len(x_com1)):
            if i!=j:
                current_distance = math.sqrt((x_com1[i]-x_com1[j])**2+(y_com1[i]- y_com1[j])**2)*130
                if current_distance <= threshold_radius[len(threshold_radius)-1]:
                    for k in range(0,len(threshold_radius)):
                        if current_distance <= threshold_radius[k]:
                            amountOfNeighbors_data1_old[i][k] = amountOfNeighbors_data1_old[i][k] + 1
                    
    e=timer()
    print("amountOfNeighbors old:", e-s)
    '''
    
    """
    s=timer()                
    amountOfNeighbors_data1 = np.zeros((len(x_com1),len(threshold_radius)),dtype=int)
    
        
        
    blockdim_1d = 32
    griddim_1d = len(amountOfNeighbors_data1)//blockdim_1d + 1
    
    amountofNeighbors_kernel[griddim_1d, blockdim_1d](d_x_com1, d_y_com1, amountOfNeighbors_data1, threshold_radius)
    
        
    
    e=timer()
    print("amountOfNeighbors new:", e-s)
    """

    #print(os.getcwd())
    #os.chdir(r"W:\users\reinhardt")
    #print(os.getcwd())

    #print(filename)
    s=timer()    

    path =  os.path.split(filename)[0] + "/AdditionalOutputs/"
    fname = path + os.path.split(filename)[1]
    np.savetxt('%s_coloc_Percentage_%s.txt' %(fname, protein), Labeling_efficiency2_to_1, delimiter= ',')
    
    #NNA = np.savetxt('%s_amount_Of_Neighbors_%s.csv' %(filename, protein), amountOfNeighbors_data1, delimiter= ',')
    NNA = np.savetxt('%s_threshold_radii.csv' %fname, threshold_radius)
    NNA = np.savetxt('%s_Neighbor_distance_%s.csv' %(fname, protein), NN_Talin_T)
    
    NNA = np.savetxt('%s_higher_Neighbors_%s.csv' %(fname, protein), higher_neighbors_Talin, delimiter= ',')
       
    e=timer()
    #print("save:", e-s)   
    #NNA.create_dataset('NNA_Data1', data=NN_Talin_T)
    #if filename2.isnumeric == False:
    #    NNA.create_dataset('NNA_Data2', data=NN_Kindlin_K)
    #    NNA.create_dataset('NNA_Data2_To_Data1', data=NN_Kindlin_Talin)
    #    NNA.create_dataset('NNA_Data1_To_Data2', data=NN_Talin_Kindlin)
    
    
    
    
    # plt.scatter(x_coords,y_coords,c=amountOfNeighbors_talin,cmap='hot')
    #plt.colorbar();
    
    #print(a_group_key)
    
    #print(len(x_com1))
    
    #print(data)
    
    end_all = timer()
    #print("total runtime:", end_all-start_all)
