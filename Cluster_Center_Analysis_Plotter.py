#!/usr/bin/env python
# coding: utf-8

#Philipp Steen 21.12.21

# In[160]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# In[161]:


#input_path_R1 = "/Volumes/pool-miblab4/users/steen/z.microscopy_raw/211129_RESI_R1-5pr_R3-3pr_R4_single_align/resi_analysis/final/R1_resi.hdf5"
#input_path_R3 = "/Volumes/pool-miblab4/users/steen/z.microscopy_raw/211129_RESI_R1-5pr_R3-3pr_R4_single_align/resi_analysis/final/R3_resi.hdf5"
input_path_R1 = "/Volumes/pool-miblab4/users/steen/z.microscopy_raw/211130_RESI_R1-5pr_2del_R3-3pr_R4_single_align/resi_analysis/final/R1_resi.hdf5"
input_path_R3 = "/Volumes/pool-miblab4/users/steen/z.microscopy_raw/211130_RESI_R1-5pr_2del_R3-3pr_R4_single_align/resi_analysis/final/R3_resi.hdf5"



R1_table = pd.read_hdf(input_path_R1, key = 'locs')
R3_table = pd.read_hdf(input_path_R3, key = 'locs')
R1_table.sort_values(by=['group', 'frame'])
R3_table.sort_values(by=['group', 'frame'])
R1_table["x"] = 130*R1_table["x"]-33240
R3_table["x"] = 130*R3_table["x"]-33240
R1_table["y"] = -(130)*R1_table["y"]+33320
R3_table["y"] = -(130)*R3_table["y"]+33320
# Groups 0, 1, 8, 9, 10 and 11 are R4
# Groups 2, 3, 4, 5, 6, 7 are R1 or R3

# Groups 0, 1, 2, 3, 10, 11 are R4
# Groups 4, 5, 6, 7, 8, 9 are R1 or R3


# In[162]:


R4_positions_R1_channel = []
R4_std_R1_channel = []
R4_positions_R3_channel = []
R4_std_R3_channel = []

#for i in (0,1,8,9,10,11):
for i in (0,1,2,3,10,11):
    R1 = (R1_table[R1_table["group"] == i])
    Avg_R1 = R1["x"].mean(), R1["y"].mean()
    Std_R1 = R1["x"].std(), R1["y"].std()
    R3 = (R3_table[R3_table["group"] == i])
    Avg_R3 = R3["x"].mean(), R3["y"].mean()
    Std_R3 = R3["x"].std(), R3["y"].std()
    
    R4_positions_R1_channel.append(Avg_R1)
    R4_std_R1_channel.append(Std_R1)
    R4_positions_R3_channel.append(Avg_R3)
    R4_std_R3_channel.append(Std_R3)
    
R4_positions_R1_channel = np.asarray(R4_positions_R1_channel)
R4_positions_R3_channel = np.asarray(R4_positions_R3_channel)
R4_std_R1_channel = np.asarray(R4_std_R1_channel)
R4_std_R3_channel = np.asarray(R4_std_R3_channel)

R4_differences = (R4_positions_R1_channel - R4_positions_R3_channel)

R4_distances = (np.sqrt(R4_differences[:,0]**2 + R4_differences[:,1]**2))

print(R4_differences)
print(R4_distances)



print("R4 from R1 channel std: ", np.mean(R4_std_R1_channel, axis=0))
print("R4 from R3 channel std: ", np.mean(R4_std_R3_channel, axis=0))


# In[166]:


R1_resi = []
R1_resi_std = []
R3_resi = []
R3_resi_std = []

#for i in (2,3,4,5,6,7):
for i in (4,5,6,7,8,9):
    R1 = (R1_table[R1_table["group"] == i])
    Avg_R1 = R1["x"].mean(), R1["y"].mean()
    Std_R1 = R1["x"].std(), R1["y"].std()
    R3 = (R3_table[R3_table["group"] == i])
    Avg_R3 = R3["x"].mean(), R3["y"].mean()
    Std_R3 = R3["x"].std(), R3["y"].std()
    
    R1_resi.append(Avg_R1)
    R1_resi_std.append(Std_R1)
    R3_resi.append(Avg_R3)
    R3_resi_std.append(Std_R3)
    
R1_resi = np.asarray(R1_resi)
R3_resi = np.asarray(R3_resi)
R1_resi_std = np.asarray(R1_resi_std)
R3_resi_std = np.asarray(R3_resi_std)

Resi_differences = (R1_resi - R3_resi)
Resi_distances = (np.sqrt(Resi_differences[:,0]**2 + Resi_differences[:,1]**2))

print(Resi_differences)
print(Resi_distances)

print("R1 std: ", np.mean(R1_resi_std, axis=0))
print("R3 std: ", np.mean(R3_resi_std, axis=0))


# In[167]:


plt.figure(figsize=(12, 12))

plt.scatter(R1_resi[:,0], R1_resi[:,1], color = "red", label = "R1")
plt.scatter(R3_resi[:,0], R3_resi[:,1], color = "blue", label = "R3")

plt.scatter(R4_positions_R1_channel[:,0], R4_positions_R1_channel[:,1], color = "maroon", label = "R4 from R1 channel")
plt.scatter(R4_positions_R3_channel[:,0], R4_positions_R3_channel[:,1], color = "darkblue", label = "R4 from R3 channel")

plt.axis('equal')


# In[165]:


R4_cloudsizes = []
for i in (0,1,8,9,10,11):
    a = R1_table[R1_table["group"]==i]
    b = R3_table[R3_table["group"]==i]
    new_group = a.append(b)
    R4_cloudsizes.append((new_group["x"].mean(), new_group["y"].mean()))

R1_3_cloudsizes = []
for i in (2,3,4,5,6,7):
    a = R1_table[R1_table["group"]==i]
    b = R3_table[R3_table["group"]==i]
    new_group = a.append(b)
    R1_3_cloudsizes.append((new_group["x"].mean(), new_group["y"].mean()))
    
R4_cloudsizes = np.asarray(R4_cloudsizes)
R1_3_cloudsizes = np.asarray(R1_3_cloudsizes)

print(R4_cloudsizes)
print(R1_3_cloudsizes)


# In[ ]:




