# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:00:59 2022

@author: reinhardt
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from os.path import dirname as up

cwd = os.getcwd()
wdir = up(cwd)
os.chdir(wdir)
import tools

path = r'W:\users\reinhardt\RESI\locmofit-avg\250122\RESI_cluster_center\analysis'

filename = 'avg_locs_all_13.hdf5'

blur_factor = 3

file = os.path.join(path,filename)


df = pd.read_hdf(file, key='locs')
print(df.keys())

print(df.head())
df['lpx'] = df['lpx']*blur_factor
df['lpy'] = df['lpy']*blur_factor

print(df.head())



new_filename = filename[:-5] + '_' + str(blur_factor) + "x_blurred.hdf5"
tools.picasso_hdf5(df, new_filename, filename, path + "/")

