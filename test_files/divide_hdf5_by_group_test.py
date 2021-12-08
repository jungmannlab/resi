# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 21:46:32 2021

@author: reinhardt
"""

##########################################################################

print('Test: divide_hdf5_by_group')

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from divide_hdf5_by_group import divide_hdf5_by_group_f

cwd = currentdir

path = cwd + '/divide_hdf5_by_group/'
fname = 'R1_apicked.hdf5'

divide_hdf5_by_group_f(path + fname)


###########################################################################