#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:16:53 2021

@author: masullo
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf

plt.close('all')

# mean_0 = np.array([-20, -10]) + np.array([500, 500])
# mean_1 = np.array([-20, 10]) + np.array([500, 500])
# mean_2 = np.array([15, 0]) + np.array([500, 500])
# mean_3 = np.array([25, 0]) + np.array([500, 500])
# mean_4 = np.array([25, 15]) + np.array([500, 500])
# mean_5 = np.array([25, -10]) + np.array([500, 500])

mean_0 = np.array([-20, -10]) 
mean_1 = np.array([-20, 10]) 
mean_2 = np.array([15, 0]) 
mean_3 = np.array([25, 0])
mean_4 = np.array([25, 15]) 
mean_5 = np.array([25, -10]) 

cov = [[1, 0], [0, 1]]

samples = 20

data_A = np.concatenate((np.random.multivariate_normal(mean_0, cov, samples),  
                        np.random.multivariate_normal(mean_1, cov, samples), 
                        np.random.multivariate_normal(mean_2, cov, samples),
                        np.random.multivariate_normal(mean_3, cov, samples),
                        np.random.multivariate_normal(mean_4, cov, samples),
                        np.random.multivariate_normal(mean_5, cov, samples)),
                        axis=0)

angle = np.pi/20
transform = tf.EuclideanTransform(rotation=angle, translation=[0, 0])

data_B = np.concatenate((np.random.multivariate_normal(mean_0, cov, samples),  
                        np.random.multivariate_normal(mean_1, cov, samples), 
                        np.random.multivariate_normal(mean_2, cov, samples),
                        np.random.multivariate_normal(mean_3, cov, samples),
                        np.random.multivariate_normal(mean_4, cov, samples),
                        np.random.multivariate_normal(mean_5, cov, samples)),
                        axis=0)

data_B = transform(data_B)

transf_estimate = tf.EuclideanTransform()
transf_estimate.estimate(data_B, data_A)
params = transf_estimate.rotation, transf_estimate.translation

print(params[1], params[0])

data_est = transf_estimate(data_B)

fig, ax = plt.subplots()

ax.plot(data_A[:, 0], data_A[:, 1], 'o', alpha=0.5, label='Channel 1')
ax.plot(data_B[:, 0], data_B[:, 1], 'o', alpha=0.5, label='Channel 2')
ax.plot(data_est[:, 0], data_est[:, 1], 'o', alpha=0.5, label='Chanel 2 - transformed')
ax.set_aspect('equal')
ax.set_xlabel('x (nm)')
ax.set_xlim([-40, 40])
ax.set_ylim([-30, 30])
ax.set_ylabel('y (nm)')

plt.legend()
plt.tight_layout()


