#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Susanne Reinhardt & Rafal Kowalewski based on the algorithm
by Thomas Schlichthaerle

Description!!!!!!!!!!            Clustering, Resi locs, 

"""


from timeit import default_timer as timer
start_all = timer()
from numba import cuda
from numba import njit as _njit


import os
import h5py
import numpy as _np
import numpy as np
import pandas as pd
import tools
from scipy.spatial import distance_matrix as _dm

@_njit 
def count_neighbors_picked(dist, radius):
	"""
	Calculates number of neighbors for each point within a given 
	radius.
	Used in clustering picked localizations.
	Parameters
	----------
	dist : np.array
		2D distance matrix
	radius : float
		Radius within which neighbors are counted
	Returns
	-------
	np.array
		1D array with number of neighbors for each point within radius
	"""

	nn = _np.zeros(dist.shape[0], dtype=_np.int32)
	for i in range(len(nn)):
		nn[i] = _np.where(dist[i] <= radius)[0].shape[0] - 1
	return nn


@_njit
def local_maxima_picked(dist, nn, radius):
	"""
	Finds which localizations are local maxima, i.e., localizations 
	with the highest number of neighbors within given radius.
	Used in clustering picked localizations.
	Parameters
	----------
	dist : np.array
		2D distance matrix
	nn : np.array
		1D array with number of neighbors for each localization
	radius : float
		Radius within which neighbors are counted	
	Returns
	-------
	np.array
		1D array with 1 if a localization is a local maximum, 
		0 otherwise	
	"""

	n = dist.shape[0]
	lm = _np.zeros(n, dtype=_np.int8)
	for i in range(n):
		for j in range(n):
			if dist[i][j] <= radius and nn[i] >= nn[j]:
				lm[i] = 1
			if dist[i][j] <= radius and nn[i] < nn[j]:
				lm[i] = 0
				break
	return lm


@_njit
def assign_to_cluster_picked(dist, lm, radius):
	"""
	Finds cluster id for each localization.
	If a localization is within radius from a local maximum, it is
	assigned to a cluster. Otherwise, it's id is 0.
	Used in clustering picked localizations.
	Parameters
	----------
	dist : np.array
		2D distance matrix
	lm : np.array
		1D array with local maxima
	radius : float
		Radius within which neighbors are counted	
	Returns
	-------
	np.array
		1D array with cluster id for each localization
	"""

	n = dist.shape[0]
	cluster_id = _np.zeros(n, dtype=_np.int32)
	for i in range(n):
		if lm[i]:
			for j in range(n):
				if dist[i][j] <= radius:
					if cluster_id[i] != 0:
						if cluster_id[j] == 0:
							cluster_id[j] = cluster_id[i]
					if cluster_id[i] == 0:
						if j == 0:
							cluster_id[i] = i + 1
						cluster_id[j] = i + 1
	return cluster_id



@_njit
def check_cluster_size(cluster_n_locs, min_locs, cluster_id):
	"""
	Filters clusters with too few localizations.
	Parameters
	----------
	cluster_n_locs : np.array
		Contains number of localizations for each cluster
	min_locs : int
		Minimum number of localizations to consider a cluster valid
	cluster_id : np.array
		Array with cluster id for each localization
	Returns
	-------
	np.array
		cluster_id after filtering
	"""

	for i in range(len(cluster_id)):
		if cluster_n_locs[cluster_id[i]] <= min_locs: # too few locs
			cluster_id[i] = 0 # id 0 means loc is not assigned to any cluster
	return cluster_id


@_njit
def rename_clusters(cluster_id, clusters):
	"""
	Reassign cluster ids after filtering (to make them consecutive)
	Parameters
	----------
	cluster_id : np.array
		Contains cluster id for each localization (after filtering)
	clusters : np.array
		Unique cluster ids
	Returns
	-------
	np.array
		Cluster ids with consecutive values
	"""

	for i in range(len(cluster_id)):
		for j in range(len(clusters)):
			if cluster_id[i] == clusters[j]:
				cluster_id[i] = j
	return cluster_id


@_njit 
def cluster_properties(cluster_id, n_clusters, frame):
	"""
	Finds cluster properties used in frame analysis.
	Returns mean frame and highest fraction of localizations within
	1/20th of whole acquisition time for each cluster.
	Parameters
	----------
	cluster_id : np.array
		Contains cluster id for each localization
	n_clusters : int
		Total number of clusters
	frame : np.array
		Frame number for each localization
	Returns
	-------
	np.array
		Mean frame for each cluster
	np.array
		Highest fraction of localizations within 1/20th of whole
		acquisition time.
	"""

	# mean frame for each cluster
	mean_frame = _np.zeros(n_clusters, dtype=_np.float32)
	# number of locs in each cluster
	n_locs_cluster = _np.zeros(n_clusters, dtype=_np.int32)
	# number of locs in each cluster in each time window (1/20th 
	# acquisition time)
	locs_in_window = _np.zeros((n_clusters, 21), dtype=_np.int32)
	# highest fraction of localizations within the time windows 
	# for each cluster
	locs_frac = _np.zeros(n_clusters, dtype=_np.float32)
	# length of the time window
	window_search = frame[-1] / 20
	for j in range(n_clusters):
		for i in range(len(cluster_id)):
			if j == cluster_id[i]:
				n_locs_cluster[j] += 1
				mean_frame[j] += frame[i]
				locs_in_window[j][int(frame[i] / window_search)] += 1
	mean_frame = mean_frame / n_locs_cluster
	for i in range(n_clusters):
		for j in range(21):
			temp = locs_in_window[i][j] / n_locs_cluster[i]
			if temp > locs_frac[i]:
				locs_frac[i] = temp
	return mean_frame, locs_frac



def find_true_clusters(mean_frame, locs_frac, n_frame):
	"""
	Performs basic frame analysis on clusters.
	Checks for "sticky events" by analyzing mean frame and the
	highest fraction of locs in 1/20th interval of acquisition time.
	Parameters
	----------
	mean_frame : np.array
		Contains mean frame for each cluster
	locs_frac : np.array
		Contains highest fraction of locs withing the time window
	n_frame : int
		Acquisition time given in frames
	Returns
	-------
	np.array
		1D array with 1 if a cluster passed the frame analysis, 
		0 otherwise
	"""

	true_cluster = _np.zeros(len(mean_frame), dtype=_np.int8)
	for i in range(len(mean_frame)):
		cond1 = locs_frac[i] < 0.8
		cond2 = mean_frame[i] < n_frame * 0.8
		cond3 = mean_frame[i] > n_frame * 0.2
		if cond1 and cond2 and cond3:
			true_cluster[i] = 1
	return true_cluster






def find_clusters_picked(dist, radius):
	"""
	Counts neighbors, finds local maxima and assigns cluster ids.
	Used in clustering picked localizations.
	Parameters
	----------
	dist : np.array
		2D distance matrix
	radius : float
		Radius within which neighbors are counted
	Returns
	-------
	np.array
		Cluster ids for each localization
	"""

	n_neighbors = count_neighbors_picked(dist, radius)
	local_max = local_maxima_picked(dist, n_neighbors, radius)
	cluster_id = assign_to_cluster_picked(dist, local_max, radius)
	return cluster_id	


def postprocess_clusters(cluster_id, min_locs, frame):
	"""
	Filters clusters for minimum number of localizations and performs 
	basic frame analysis to filter out "sticky events".
	Parameters
	----------
	cluster_id : np.array
		Contains cluster id for each localization (before filtering)
	min_locs : int
		Minimum number of localizations in a cluster
	frame : np.array
		Frame number for each localization
	Returns
	-------
	np.array
		Contains cluster id for each localization
	np.array
		Specifies if a given cluster passed the frame analysis
	"""
	cluster_n_locs = _np.bincount(cluster_id) # number of locs in each cluster
	cluster_id = check_cluster_size(cluster_n_locs, min_locs, cluster_id)
	clusters = _np.unique(cluster_id)
	cluster_id = rename_clusters(cluster_id, clusters)
	n_clusters = len(clusters)
	mean_frame, locs_frac = cluster_properties(
		cluster_id, n_clusters, frame
	)
	n_frame = _np.int32(_np.max(frame))
	true_cluster = find_true_clusters(mean_frame, locs_frac, n_frame)
	return cluster_id, true_cluster


def get_labels(cluster_id, true_cluster):
	"""
	Gives labels compatible with scikit-learn style, i.e., -1 means
	a point (localization) was not assigned to any cluster
	Parameters
	----------
	cluster_id : np.array
		Contains cluster id for each localization
	true_cluster : np.array
		Specifies if a given cluster passed the frame analysis
	Returns
	-------
	np.array
		Contains label for each localization
	"""

	labels = -1 * _np.ones(len(cluster_id), dtype=_np.int32)
	for i in range(len(cluster_id)):
		if cluster_id[i] != 0 and true_cluster[cluster_id[i]] == 1:
			labels[i] = cluster_id[i] - 1
	return labels




def clusterer_picked_2D(x, y, frame, radius, min_locs):
	"""
	Clusters picked localizations while storing distance matrix and 
	returns labels for each localization (2D).
	Works most efficiently if less than 700 locs are provided.
	Parameters
	----------
	x : np.array
		x coordinates of picked localizations
	y : np.array
		y coordinates of picked localizations
	frame : np.array
		Frame number for each localization
	radius : float
		Clustering radius
	min_locs : int
		Minimum number of localizations in a cluster
	Returns
	-------
	np.array
		Labels for each localization
	"""

	xy = _np.stack((x, y)).T
	dist = _dm(xy, xy) # calculate distance matrix
	cluster_id = find_clusters_picked(dist, radius)
	cluster_id, true_cluster = postprocess_clusters(
		cluster_id, min_locs, frame
	)
	return get_labels(cluster_id, true_cluster)




def clusterer_picked_3D(x, y, z, frame, radius_xy, radius_z, min_locs):
	"""
	Clusters picked localizations while storing distance matrix and 
	returns labels for each localization (3D).
	z coordinate is scaled to account for different clustering radius
	in z.
	Works most efficiently if less than 600 locs are provided.
	Parameters
	----------
	x : np.array
		x coordinates of picked localizations
	y : np.array
		y coordinates of picked localizations
	frame : np.array
		Frame number for each localization
	z : np.array
		z coordinates of picked localizations
	radius_xy : float
		Clustering radius in x and y directions
	radius_z : float
		Clustering radius in z direction
	min_locs : int
		Minimum number of localizations in a cluster
	Returns
	-------
	np.array
		Labels for each localization
	"""

	xyz = _np.stack((x, y, z * (radius_xy/radius_z))).T # scale z
	dist = _dm(xyz, xyz)
	cluster_id = find_clusters_picked(dist, radius_xy)
	cluster_id, true_cluster = postprocess_clusters(
		cluster_id, min_locs, frame
	)
	return get_labels(cluster_id, true_cluster)






def save_locs_in_cluster(cluster_df, hdf5_file, radius_xy, min_locs, flag_3D, flag_group_input, radius_z = 0):
    '''
    Saves the localizations that were assigned to a cluster into a new
    Picasso hdf5 file. 

    Parameters
    ----------
    cluster_df : dataframe
        contains the localizations including the group column assigning them 
        to a cluster.
    hdf5_file : string
        filenmae of the original file. Used for new filename.
    radius_xy : float/int
        xy-radius used clustering. Used for new filename.
    min_locs : int
        Minimal number of locs a cluster needed to have. Used for new filename.
    flag_3D : bool
        True if 3D dataset.
    flag_group_input : bool
        True if original dataset already had a group column which is now saved
        as a column named group_input
    radius_z : float/int
        z-radius used clustering. Used for new filename.
    '''
  
    if not flag_3D:
        if not flag_group_input:

            cluster_df = cluster_df.astype({'frame': 'u4', 'x': 'f4', 'y': 'f4', 'photons': 'f4', 
                            'sx': 'f4', 'sy': 'f4', 'bg': 'f4', 'lpx': 'f4','lpy': 'f4',
                            'ellipticity': 'f4', 'net_gradient': 'f4', 'group': 'u4'})
            df2 = cluster_df.reindex(columns = ['frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 
                                        'lpx', 'lpy', 'ellipticity', 'net_gradient', 'group'], fill_value=1)
           
        else:

            cluster_df = cluster_df.astype({'frame': 'u4', 'x': 'f4', 'y': 'f4', 'photons': 'f4', 
                            'sx': 'f4', 'sy': 'f4', 'bg': 'f4', 'lpx': 'f4','lpy': 'f4',
                            'ellipticity': 'f4', 'net_gradient': 'f4', 'group_input': 'u4', 'group': 'u4'})
            df2 = cluster_df.reindex(columns = ['frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 
                                        'lpx', 'lpy', 'ellipticity', 'net_gradient', 'group_input', 'group'], fill_value=1)

        
    else:
        if not flag_group_input:

            cluster_df = cluster_df.astype({'frame': 'u4', 'x': 'f4', 'y': 'f4', 'z': 'f4', 'photons': 'f4',
                            'sx': 'f4', 'sy': 'f4', 'bg': 'f4', 'lpx': 'f4','lpy': 'f4',
                            'ellipticity': 'f4', 'net_gradient': 'f4', 'd_zcalib': 'f4', 'group': 'u4'})
            df2 = cluster_df.reindex(columns = ['frame', 'x', 'y', 'z', 'photons', 'sx', 
                                        'sy', 'bg', 'lpx', 'lpy', 'ellipticity', 'net_gradient', 'd_zcalib', 'group'], fill_value=1)

        else:

            cluster_df = cluster_df.astype({'frame': 'u4', 'x': 'f4', 'y': 'f4', 'z': 'f4', 'photons': 'f4',
                            'sx': 'f4', 'sy': 'f4', 'bg': 'f4', 'lpx': 'f4','lpy': 'f4',
                            'ellipticity': 'f4', 'net_gradient': 'f4', 'd_zcalib': 'f4', 'group_input': 'u4', 'group': 'u4'})
            df2 = cluster_df.reindex(columns = ['frame', 'x', 'y', 'z', 'photons', 'sx', 
                                        'sy', 'bg', 'lpx', 'lpy', 'ellipticity', 'net_gradient', 'd_zcalib', 'group_input', 'group'], fill_value=1)

    path = os.path.split(hdf5_file)[0] + "/"
    filename_old = os.path.split(hdf5_file)[1]
    if not flag_3D:
        filename_new = '%s_ClusterD%s_%d.hdf5' % (filename_old[:-5], str(radius_xy), min_locs)
    else:
        filename_new = '%s_ClusterD%s_%d_%s.hdf5' %(filename_old[:-5], str(radius_xy), min_locs, str(radius_z))
    tools.picasso_hdf5(df2, filename_new, filename_old, path)
    
    
    

def wtd_sed_sums(group, x, lpx):
    '''
    Calculates the sums in the formula for the weighted standard error of the mean.

    Parameters
    ----------
    group : grouped dataframe
        This dataframe contains all localizations that were assigned to a cluster. 
        It is grouped by the cluster identity saved in the 'group' column.
    x : string
        Key to extract the respective coordinate from the dataframe. 
        Either 'x' or 'y'.
    lpx : string
        Key to extract the localization precision from the dataframe.
        Either 'lpx' or 'lpy'

    Returns
    -------
    pandas series
        Sum in the formula for the weighted standard error of the mean.
    '''
    x = group[x]
    w = 1/group[lpx]**2
    
    return (w * (x - (w * x).sum() / w.sum())**2).sum() / w.sum()


def RESI_locs(grouped, flag_3D):
    '''
    Calculates the weighted mean for each cluster (RESI localization) as
    well as the weighted standard error of the mean. 
    The localization precisions in x and y direction serve as weights, 
    in z direction no weights are available and the unweighted average is
    calculated.

    Parameters
    ----------
    cluster_df : dataframe
        This dataframe contains all localizations that were assigned to a cluster. 
        It is grouped by the cluster identity saved in the 'group' column.
    flag_3D : bool
        True if 3D dataset. False if 2D dataset.

    Returns
    -------
    several pandas series
        One pandas series containing the RESI localization for each direction, 
        one for the number of DNA-PAINT locs per cluster and one for the weighted 
        standard error of the mean. 

    '''


    

    # RESI localization: Weighted mean of the DNA-PAINT localizations in each
    # detected cluster. 
    # Weights for x and y average: the inverse squared localization precision.
    # No weights for z average as no localization precision in z is available.
    x_av_wtd = grouped.apply(lambda x: np.average(x['x'],weights=1/x['lpx']/x['lpx']))
    x_av_wtd.name = "x_av_wtd"
    y_av_wtd  = grouped.apply(lambda x: np.average(x['y'],weights=1/x['lpy']/x['lpy']))
    y_av_wtd.name = "y_av_wtd"
    if flag_3D:
        z_av_wtd = grouped.apply(lambda x: np.average(x['z'])) #in grouped, which is based on df2 the z_coordinates are in nm
        z_av_wtd.name = "z_av_wtd"
    group_size = grouped.size()
    group_size.name = "group_size"

    
    # Weighted standard error of the mean for each RESI localization.
    sed_x = np.sqrt(grouped.apply(wtd_sed_sums, "x", "lpx")/(group_size - 1))
    sed_y = np.sqrt(grouped.apply(wtd_sed_sums, "y", "lpy")/(group_size - 1))
    sed_xy = np.mean([sed_x, sed_y], axis = 0)
    
    if flag_3D:
        return x_av_wtd, y_av_wtd, z_av_wtd, group_size, sed_xy
    else:
        return x_av_wtd, y_av_wtd, group_size, sed_xy




def save_RESI_locs(group_means, hdf5_file, radius_xy, min_locs, flag_3D, radius_z = 0):
    '''
    Saves the RESI localizations and their respective standard error of the
    means in a new Picasso hdf5 file.
    Picasso Render can render every RESI localizations with a Gaussian blur
    with sigma being the standard error of the mean

    Parameters
    ----------
    group_means : dataframe
        contains the RESI localizations and their weighted standard errors of 
        the mean. All other columns contain the mean values taken over the 
        DNA-PAINT localizations belonging to this RESI loccalizations cluster.
    hdf5_file : string
        filenmae of the original file. Used for new filename.
    radius_xy : float/int
        xy-radius used clustering. Used for new filename.
    min_locs : int
        Minimal number of locs a cluster needed to have. Used for new filename.
    flag_3D : bool
        True if 3D dataset.
    radius_z : float/int
        z-radius used clustering. Used for new filename.
    '''    

    if not flag_3D:

        group_means = group_means.astype({'frame': 'u4', 'x': 'f4', 'y': 'f4', 
                        'photons': 'f4', 'sx': 'f4', 'sy': 'f4',
                        'bg': 'f4', 'lpx': 'f4','lpy': 'f4',
                        'group': 'u4', 'n': 'u4'})
        df3 = group_means.reindex(columns = ['frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy', 'group', 'n'], fill_value=1)

    else:

        group_means = group_means.astype({'frame': 'u4', 'x': 'f4', 'y': 'f4', 'z': 'f4', 
                        'photons': 'f4', 'sx': 'f4', 'sy': 'f4',
                        'bg': 'f4', 'lpx': 'f4','lpy': 'f4', 'lpz': 'f4',
                        'group': 'u4', 'n': 'u4'})
        df3 = group_means.reindex(columns = ['frame', 'x', 'y', 'z', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy', 'lpz', 'group', 'n'], fill_value=1)


    path = os.path.split(hdf5_file)[0] + "/"
    filename_old = os.path.split(hdf5_file)[1]
    if not flag_3D:
        filename_new = '%s_resi_%s_%d.hdf5' % (filename_old[:-5], str(radius_xy), min_locs)
    else:
        filename_new = '%s_resi_%s_%d_%s.hdf5' %(filename_old[:-5], str(radius_xy), min_locs, str(radius_z))
    tools.picasso_hdf5(df3, filename_new, filename_old, path)




def clusterer_start(hdf5_file, radius_xy, min_locs, px_size, radius_z):
    '''
    

    Parameters
    ----------
    hdf5_file : TYPE
        DESCRIPTION.
    radius_xy : TYPE
        DESCRIPTION.
    min_locs : TYPE
        DESCRIPTION.
    px_size : TYPE
        DESCRIPTION.
    radius_z : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    # Read in HDF5-File
    data_df = pd.read_hdf(hdf5_file, key = 'locs')   

    # Convert radii from nm to pixel
    radius_xy_px = radius_xy / px_size
    radius_z_px = radius_z / px_size


    x_coords = np.array(data_df['x'])
    y_coords = np.array(data_df['y'])
    frame = np.array(data_df['frame'])

    print()
    print("number of coordinates:", len(x_coords))

    # Check if it is a 2d or 3d dataset
    try:
        z_coords = data_df['z']/px_size
    except ValueError:
        print("2D data recognized.")
        flag_3D = False
    else:
        print("3D data recognized.")
        if radius_z == 0: # 2D
            flag_3D = False
            print("No z-radius specified. Clustering will be performed based on xy coordinates only.")
        else: #3D
            flag_3D = True
            z_coords = np.array(data_df['z'])/px_size
            
            
            
    # keep group info as group_input if already present
    # group column will be overwritten by Cluster ID
    try: 
        data_df['group_input'] = data_df['group']
    except ValueError:
        print("No group information detected in input file.")
        flag_group_input = False
    else:
        print("Group information detected in input file. \nThis input group column will be kept in the 'group_input' column. \n The new 'group' column describes the clusters.")
        flag_group_input = True
    
    
    # Start clustering procedure
    if flag_3D: # 3D
        labels = clusterer_picked_3D(
            x_coords,
            y_coords,
            z_coords,
            frame,
            radius_xy_px,
            radius_z_px,
            min_locs,
        )
    else:
        labels = clusterer_picked_2D(
            x_coords,
            y_coords,
            frame,
            radius_xy_px,
            min_locs,
        )
    
    cluster_df = data_df.copy()
    cluster_df['group'] = labels
    cluster_df.drop(cluster_df[cluster_df.group == -1].index, inplace = True)
    cluster_df.reset_index(drop = True)
    #temp_locs = temp_locs[temp_locs.group != -1]
    
    
    if flag_3D:
        save_locs_in_cluster(cluster_df, hdf5_file, radius_xy, min_locs, flag_3D, flag_group_input, radius_z)
    else:
        save_locs_in_cluster(cluster_df, hdf5_file, radius_xy, min_locs, flag_3D, flag_group_input)




    # Calculate RESI localizations and the standard error of the mean
    grouped = cluster_df.groupby("group")
    if flag_3D:
        x_av_wtd, y_av_wtd, z_av_wtd, group_size, sed_xy = RESI_locs(grouped, flag_3D)
    else:
        x_av_wtd, y_av_wtd, group_size, sed_xy = RESI_locs(grouped, flag_3D)
    

    # Mean value for each column in cluster_df dataframe on a per-group basis  
    group_means = grouped.mean()

    # In the averaged dataframe replace the coordinate columns and the 
    # lpx/lpy/lpz columns with the weighted mean and the (weighted) standard
    # error of the mean respectively
    group_means['x'] = x_av_wtd
    group_means['y'] = y_av_wtd
    if flag_3D:
        group_means['z'] = z_av_wtd
    # Saving the standard error of the mean in the localization precision columns
    # in the Picasso hdf5 allows Picasso Render to display a gaussian blur around
    # each RESI localization that corresponds to the standard error of the mean.
    group_means['lpx'] = sed_xy
    group_means['lpy'] = sed_xy
    if flag_3D:
        # Approximate the uncertainty in z as two times the xy standard error
        # of the mean.
        # Note that Picasso Render does not take the lpz column as input 
        # but calculates lpz on its own from sed_xy in the same way.
        group_means['lpz'] = 2 * sed_xy
    # Save group ID of origami, not of clusters in origami in the group column
    group_means['group'] = group_means['group_input']
    # Number of locs per cluster
    group_means['n'] = group_size


    #group_means['group'] = np.full(shape = len(group_means), fill_value = np.mean(np.array(data_df['group']))) 
    
    # Get rid of columns we don't need
    group_means = group_means.drop(columns=['ellipticity', 'net_gradient', 'd_zcalib', 'group_input'])

    if flag_3D:
        save_RESI_locs(group_means, hdf5_file, radius_xy, min_locs, flag_3D, radius_z)
    else:
        save_RESI_locs(group_means, hdf5_file, radius_xy, min_locs, flag_3D)


    '''
    try:
        os.mkdir(os.path.split(hdf5_file)[0] + '/AdditionalOutputs')
    except OSError:
        print ("AdditionalOutputs folder already exists")
    '''
    
    print('group keys', group_means.keys())
    
    
    
   

"""


    
    '''
    Save npz with arrays used for postprocessing
    '''
    if not flag_3D:
        np.savez('%s_varsD%s_%d' %(filename[:-5], threshold_radius_str, cluster_size_threshold),
             data2_x=data2_x,data2_y=data2_y, data2_frames=data2_frames, data2_group=data2_group,
             new_com_x_cluster=data3_x, new_com_y_cluster=data3_y,
             amountOfNeighbors_data=amountOfNeighbors_data, x_coords=x_coords, y_coords=y_coords) 
    else:
        np.savez('%s_varsD%s_%d_%s' %(filename[:-5], threshold_radius_str, cluster_size_threshold, str(radius_z)),
             data2_x=data2_x,data2_y=data2_y, data2_z=data2_z, data2_frames=data2_frames, data2_group=data2_group,
             new_com_x_cluster=data3_x, new_com_y_cluster=data3_y,
             new_com_z_cluster=data3_z_pl, amountOfNeighbors_data=amountOfNeighbors_data, 
             x_coords=x_coords, y_coords=y_coords, z_coords=z_coords)
"""