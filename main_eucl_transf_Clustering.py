# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:01:03 2020

@author: reinhardt
"""

import sys
import numpy as np
import glob
import os
import os.path
import itertools
import pandas as pd


from MainFunctions.Clusterer_resi_f import clusterer_resi
from MainFunctions.Cluster_PostProcessing_new_f import postprocessing
from MainFunctions.Cluster_PostProcessing_new_f import postprocessing_cross
from MainFunctions.find_eucl_transf_f import find_eucl_transf_f
from MainFunctions.apply_eucl_transf_f import apply_eucl_transf_f
'''Parameters'''
'''============================================================================'''


'''Please copy and paste the path to the folder containing the data that should be analyzed.'''
path = r"W:\users\reinhardt\z.software\Git\RESI\RESI\test_files\main_eucl_transf_Clustering_3d"
# for testfiles: path = r"W:\users\reinhardt\z.software\Git\RESI\RESI\test_files\main_eucl_transf_Clustering"

'''Please choose a value for the following two parameters.'''
colocalization_radius = 25  # (in nm) This is the distance necessary for the Postprocessing script. 
                            # It is only used for calculating the percentage of molecules that have another molecule within this distance.
                            # Thus, this is the distance when 2 molecules are considered to be neighbors (no matter if within one protein species or across two species) 


'''Please specify which hdf5 files you want to analyze as well as the respective parameters. Please follow the following pattern'''

data = [["R1", "R1_apicked", 11, 50],
        ["R3", "R3_apicked", 11, 50]]


radius_z = 22 # set to 0 for 2d data


'''
for testfiles use: 
data = [["R1", "R1_apicked", 4, 50],
        ["R3", "R3_apicked", 4, 50]]

"""

"""
data = [["Protein1", "filename_base1", radius1, min_cluster_size1, olig_inter_protein_distance1],
        ["Protein2", "filename_base2", radius2, min_cluster_size2, olig_inter_protein_distance2]]

# Protein1 is an ideally short name for the protein imaged in this file.
# filename_base1: All hdf5 files that have this string in their filename are considered to be clustered. 
    # If for example several pick files have to be analyzed the filename_base should only contain the part of the filename that these pick files have in common.
    # The code will automatically exclude hdf5 files which contain the word "ClusterD" or "coupling" because these are output files of the here performed anlysis. 
    # Note that every file that should be clustered needs to contain the word "picked". Otherwise it will not be clustered.
# radius1 is the radius used for Clustering.
# min_cluster_size1: This is N_min - the minimal number of localizations that a localization cloud has to contain to be considered a true cluster.
# olig_inter_protein_distance1: This is the step size used for defining oligomers. Two proteins inside a cluster must be reached within steps of this size.
'''




'''The script'''
'''============================================================================'''

'''Find euclidian transformation.'''
'''============================================================================'''
# This code expects two different channels!
if len(data) != 2:
    raise Exception("There must be data from exactly two channels!")

# Before running the code for finding the transformation, check if the output file
# eucl_transf_data.xlsx already exists from a previous run of the code.

eucl_transf_data = os.path.join(path,"eucl_transf/eucl_transf_data.xlsx")
if os.path.isfile(eucl_transf_data) != True: 
    print("Find best Euclidian transformation for channel alignment")
    # The transformation will be performed on the R4 sites in channel1 and channel 3
    path_alignment_picks = os.path.join(path,"alignment_picks")

    ch13_files = glob.glob(os.path.join(path_alignment_picks, "*.hdf5"))
    ch1_files = sorted(file for file in ch13_files if data[0][1] in file)
    print('ch13_files', ch13_files)
    print('ch1_files', ch1_files)
    # get the respective list for the ch3 files. 
    # Instead of extracting it in the same way from the ch13_files list
    # it will be created from the ch1_files list. If it would be extracted
    # from ch3_files, we might not notice if files in ch3 are missing.... 
    ch3_files = []
    for ch1_file in ch1_files:
        ch3_file = ch1_file.replace(data[0][1], data[1][1])
        ch3_files.append(ch3_file)
        #print("1", ch1_file)
        #print("3", ch3_file)

    find_eucl_transf_f(path, ch1_files, ch3_files)

else:
    print("Euclidian transformation for channel alignment has already been determined.")


'''Apply euclidian transformation to R3 channel.'''
'''============================================================================'''
# Before running the code for finding the transformation, check if the output file
# eucl_transf_data.xlsx already exists from a previous run of the code.

eucl_transf_data = os.path.join(path,"eucl_transf/eucl_transf_data.xlsx")
if os.path.isfile(eucl_transf_data) == True:
    ch13_files = glob.glob(os.path.join(path, "*.hdf5"))
    ch1_files = sorted(file for file in ch13_files if data[0][1] in file and "ori" in file and "ClusterD" not in file and "_resi_" not in file)

    ch3_files = []
    for ch1_file in ch1_files:
        ch3_file = ch1_file.replace(data[0][1], data[1][1])
        ch3_files.append(ch3_file)
        #print("1", ch1_file)
        #print("3", ch3_file)

    check_aligned_file = os.path.split(ch3_files[-1])[1]
    check_aligned_file = os.path.split(ch3_files[-1])[0] + "/" + check_aligned_file[:-5] + "_aligned.hdf5"
    if os.path.isfile(check_aligned_file) != True:
        apply_eucl_transf_f(path, ch1_files, ch3_files)

else:
    raise Exception("Euclidian transformation for channel alignment has not yet been determined.")



'''Perform Clustering'''
'''============================================================================'''

def Clusterer_check(path, radius, min_cluster_size, filename_base, i):
    Clusterer_filename_extension = "_ClusterD"+str(radius)+"_"+str(min_cluster_size)
    for file in glob.glob(os.path.join(path, "*.hdf5")): # searches all hdf5 files
        
        if filename_base in file and "picked" in file and Clusterer_filename_extension not in file and "ClusterD" not in file and "coupling" not in file and "_resi_" not in file and "ori" in file: 
            if (i == 0) or (i == 1 and "_aligned" in file):
            # check if file is part of the cell to be clustered (filename base in file)
            # do not apply clusterer to whole cell ("picked" in file)
            # check that the file itself is not a file produced by the clusterer itself (file does not contain ClusterD6_15 etc.)
            
                npz_file = file[:-5] + "_varsD" + str(radius) + "_" + str(min_cluster_size) + ".npz"
                if os.path.isfile(npz_file) != True: 
                    # check that the file has not been clustered already
                    #print("clusterer starts the file", file)
                    clusterer_resi(file, radius, min_cluster_size, radius_z)
                    #print("clusterer has finished the file", file)
                    #print() 
                
'''Clusterer'''
for i in range(len(data)):
    protein_info = data[i]
    radius = protein_info[2]
    min_cluster_size = protein_info[3]
    filename_base = protein_info[1]
        
    Clusterer_check(path, radius, min_cluster_size, filename_base, i)

print("clusterer finished")  



'''Postprocessing'''
'''============================================================================'''

for protein_info in data:
    protein = protein_info[0]
    radius = protein_info[2]
    min_cluster_size = protein_info[3]
    filename_base = protein_info[1]
    
    #print(glob.glob(os.path.join(path, "*.npz")))
    for file_npz in glob.glob(os.path.join(path, "*.npz")):
        if filename_base in file_npz:

            postprocess_hNN_file = os.path.split(file_npz)[0] + "/AdditionalOutputs/" + os.path.split(file_npz)[1] + "_higher_Neighbors_" + protein + ".csv"
            if os.path.isfile(postprocess_hNN_file) != True:
    
                postprocessing(protein, file_npz, colocalization_radius)


# cross correlation
if len(data) > 1:
    for pair in itertools.combinations(data,2): # pair = all combinations of proteins, each combination stored as a tuple of two elements in the data list

        protein1 = pair[0][0]
        radius1 = pair[0][2]
        min_cluster_size1 = pair[0][3]
        filename_base1 = pair[0][1]
        
        protein2 = pair[1][0]
        radius2 = pair[1][2]
        min_cluster_size2 = pair[1][3]
        filename_base2 = pair[1][1]
        
        for file_npz1 in glob.glob(os.path.join(path, "*.npz")):
            if filename_base1 in file_npz1 and "_varsD"+str(radius1)+"_"+str(min_cluster_size1) in file_npz1: # data 1 is from filename_base1 and the corresponding data2 file will be searched automatically
                file_npz2 = file_npz1.replace(filename_base1, filename_base2)
                file_npz2 = file_npz2.replace("_varsD"+str(radius1)+"_"+str(min_cluster_size1), "_aligned_varsD"+str(radius2)+"_"+str(min_cluster_size2))
                #print("npz 1", file_npz1)
                #print("npz 2", file_npz2)
                
                resi_file1 = file_npz1.replace("varsD", "resi_")
                resi_file1 = resi_file1.replace("npz", "hdf5")
                #print("resi file1", resi_file1)
                resi_file2 = file_npz2.replace("varsD", "resi_")
                resi_file2 = resi_file2.replace("npz", "hdf5")
        
                postprocess_hNN_file_ex_1 = os.path.split(file_npz1)[0] + "/AdditionalOutputs/" + os.path.split(file_npz1)[1] + "_higher_Neighbors_" + protein1 + "_to_" + protein2 + ".csv"
                postprocess_hNN_file_ex_2 = os.path.split(file_npz2)[0] + "/AdditionalOutputs/" + os.path.split(file_npz2)[1] + "_higher_Neighbors_" + protein2 + "_to_" + protein1 + ".csv"

                if os.path.isfile(postprocess_hNN_file_ex_1) != True or os.path.isfile(postprocess_hNN_file_ex_2) != True:
                    postprocessing_cross(protein1, protein2, file_npz1, file_npz2, resi_file1, resi_file2, colocalization_radius)
                    
    
print("postprocessing finished")
