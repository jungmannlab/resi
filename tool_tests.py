#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:20:37 2021

@author: Luciano A. Masullo

This script contains test and examples of how to use the functions in tools.py

"""

##########################################################################

print('Test 1 - picasso_hdf5')

import os
import pandas as pd 
import tools

cwd = os.getcwd()

path = cwd + '/test_files/'
fname = 'R3_origami_data.hdf5'

my_df = pd.read_hdf(path + fname, key='locs') 

### for example, convert loc coordinates to nm 

px_size = 130 # nm
my_df.x = my_df.x * px_size
my_df.y = my_df.y * px_size

new_name = 'R3_origami_data_processed.hdf5'

tools.picasso_hdf5(df=my_df, hdf5_fname=new_name, hdf5_oldname=fname, path=path)

###########################################################################