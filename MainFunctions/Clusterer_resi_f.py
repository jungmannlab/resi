
from timeit import default_timer as timer
start_all = timer()
from numba import cuda


import os
import h5py
import numpy as np
import sys
#import seaborn as sns
#from sklearn.cluster import DBSCAN
import math
import tools



def clusterer_resi(hdf5_file, radius, min_cluster_size):

    #Read in HDF5-File
    
    
    #filename = sys.argv[1]
    filename = hdf5_file
    cluster_size_threshold = min_cluster_size

    
    f1 = h5py.File(filename, 'r')
    a_group_key = list(f1.keys())[0]
    data = np.array(f1[a_group_key])
    

    x_coords = np.ascontiguousarray(data['x'])
    y_coords = np.ascontiguousarray(data['y'])
    frame = np.ascontiguousarray(data['frame'])
    print("number of coordinates:", len(x_coords))

    threshold_radius_str = str(radius)
    threshold_radius = float(radius)

    
    
    '''
    =============================================================================
    1) Calculate amount of neighbors within the threshold radius around every localization 
    =============================================================================
    '''
    ''' Now the amount of neighbors is calculated inside a cuda kernel, where each thread corresponds
        to one localization. Inside the kernel, each thread loops again over all localizations to 
        find the localizations that are within the readius around the localization centered in this thread.
    '''
    
    start2 = timer()
    
    #check for each cluster how many neighbors it has within this distance
    amountOfNeighbors_data = np.zeros(len(x_coords), dtype=np.int16)
    # datatype can be even less consuming as it only takes values 0 and 1
    
    
    #transfer the coordinates to the devide (GPU) only once, as they will be used by several kernels.
    d_x_coords = cuda.to_device(x_coords)
    d_y_coords = cuda.to_device(y_coords)
    #s=timer()
    d_amountOfNeighbors_data = cuda.to_device(amountOfNeighbors_data)
    #e=timer()
    #print()
    #print("to device amount of neighbors (16)", e-s)
    
    
    
    # calculates for each coordinate the amount of neighbours within the entered radius
    @cuda.jit
    def amountOfNeighbors_kernel_new(amountOfNeighbors_data, threshold_radius, x_coords, y_coords):
        i = cuda.grid(1) # i = absolute thread position
        
        
        if (i >= len(x_coords)):
            return
    
        sumtemp = 0
        
        for j in range(len(x_coords)):
            current_distance = math.sqrt((x_coords[i]-x_coords[j])**2+(y_coords[i]-y_coords[j])**2)*130
    
            if current_distance <= threshold_radius:
                sumtemp += 1
        
        amountOfNeighbors_data[i] = sumtemp -1 # the -1 replaces the condition if i!=j. Testing this vor every element in x_coords costs time and the result what happens if i = j is always 0. 
        # so we can just reduce the amount of neighbors by 1 to get the real amount of neighbors without counting the distance of i to itself.
         
                    
    # call the kernel
    blockdim_1d = 32
    griddim_1d = len(x_coords)//blockdim_1d + 1
    amountOfNeighbors_kernel_new[griddim_1d, blockdim_1d](d_amountOfNeighbors_data, threshold_radius, d_x_coords, d_y_coords)
    cuda.synchronize()
    
    
    #amountOfNeighbors_data = d_amountOfNeighbors_data.copy_to_host()
    end2 = timer()
    #print()
    #print("amount of neighbors", end2-start2)
    #print()
    
    
    
    '''
    =============================================================================
    1b) find local maxima in distance r
    =============================================================================
    '''
    
    ''' Now the amount of neighbors is calculated inside a cuda kernel, where each thread corresponds
        to one localization. Inside the kernel, each thread loops again over all localizations to 
        find out whether there is another localizations within the readius that has more neighbors 
        than the localization centered in this thread.
    '''
    # description how it works:
    # compare the amount of neighbors within the threshold_radius of a coordinate with the respective amount of all other coordinates in the same box that are closer than the threshold distance.
    # f.ex. compare i to j: i has more neighbors than j -> set local_maxima_data[i] to 1
    # then compare i to k: i has less neighbors than k: -> set local_maxima_data[i] to 0 and exit the second loop and start with a new coordinate to compare to all the other ones in its box
    # if a coordinate gets assigned a 0, it is definetly not the local maximum in the box
    # this should end with one coordinate assigned to 1 and all others to 0 (only exception: some coordinates within the same box have exactly the same number of neighbors)
    
    
    local_maxima_data = np.zeros(len(x_coords), dtype=np.int8)
    #print("before datatransfer")
    #s=timer()
    d_local_maxima_data = cuda.to_device(local_maxima_data)
    #e=timer()
    #print("to device local_maxima_data (8)", e-s)
    
    
    @cuda.jit
    def local_maxima_kernel(local_maxima_data, amountOfNeighbors_data, threshold_radius, x_coords, y_coords):
    
        i = cuda.grid(1) # i = absolute thread position
        
        
        if (i >= len(x_coords)):
            return
    
        
        for j in range(0, len(x_coords)):       
            current_distance = math.sqrt((x_coords[i]-x_coords[j])**2+(y_coords[i]-y_coords[j])**2)*130
            
            if (current_distance <= threshold_radius) and (amountOfNeighbors_data[i] >= amountOfNeighbors_data[j]):
                local_maxima_data[i] = 1
            if (current_distance <= threshold_radius) and (amountOfNeighbors_data[i] < amountOfNeighbors_data[j]):
                local_maxima_data[i] = 0
                break # contrary to the non GPU code the break statement does not speed up the code significantly (probably due to the organizations in warps). 
                      # however it is important to have it because then we make sure that local_maxima_data[i] really is 0 if it occurs once. Without the break
                      # the for loop continues and the 0 can be changed to 1 in the next round. And if by chance in no further round it is set to 0 we will get a wrong result.
    
    # Within the given radius only one cluster center can exist (with value local_maxima_data[i]=1. However, if there are two locs with exactly the same
    # amount of neighbors they will both have the value 1
    
    start3 = timer()
    
                    
    # call the kernel
    blockdim_1d = 32
    griddim_1d = len(x_coords)//blockdim_1d + 1
    local_maxima_kernel[griddim_1d, blockdim_1d](d_local_maxima_data, d_amountOfNeighbors_data, threshold_radius, d_x_coords, d_y_coords)
    cuda.synchronize()
    end3 = timer()
    #print("local maxima kernel ex", end3-start3)
    #print()
    
    local_maxima_data = d_local_maxima_data.copy_to_host()
    #m = timer()
    #print("to host local_maxima_data", m-end3)
    #print(local_maxima_data)
    
    
    
    '''
    define clusters according to maxima
    
    '''
    ''' Note that the following kernel also changes how the clusters are named and thus how the content of in cluster_Nr_data 
        will look like compared to the CPU version. This simplified the implementation of the kernel. 
        In the kernel cluster_Nr_data_rename_kernel the numbering of clusters will be translated to how it is in the CPU clusterer.
    '''
    
    
    # for each localization it will contain the information to which cluster it belongs.
    # if the coordinate at index j is in the cluster of the localization i, which is the cluster center of the cluster, then 
    # the value of cluster_Nr_data[j] will be i+1
    cluster_Nr_data = np.zeros(len(x_coords), dtype=np.int32)
    current_distance = 1000
    
    
    assignment_radius = threshold_radius
    

    @cuda.jit
    def assign_to_cluster_kernel(i, assignment_radius, cluster_Nr_data, x_coords, y_coords):
        j = cuda.grid(1)
        
      
        if (j >= len(x_coords)):
            return
    
        
        current_distance = math.sqrt((x_coords[i]-x_coords[j])**2+(y_coords[i]-y_coords[j])**2)*130
        
        if current_distance <= assignment_radius:
            # The loop that calls this kernel launches it with i being one of the cluster centers found previously.
            # Thus, cluster_NR_data[i] will be equal to i+1 (i belongs to its own cluster of course). (Explanation for why we set it to i+1 and not i is below the kernel)
            # To set the array element i to its value i+1 we check first if the cluster center i was already assigned previously to another cluster (cluster_Nr_data[i]!= 0)
            # which can happen if there is another cluster center within the radius that has exactly the same number in the amountOfNeighbors_data array
            # If this is not the case we can assign it to its own cluster (value i+1)
            # Because we want that the value of the cluster center i is only set once and not in every thread (writing to one variable from several threads is risky (can produce different outcomes))
            # we do this specifically in thread 1. 
            
            if (cluster_Nr_data[i] != 0):
                if (cluster_Nr_data[j] == 0):
                    cluster_Nr_data[j] = cluster_Nr_data[i]
            if (cluster_Nr_data[i] == 0): 
                if j == 0: 
                    cluster_Nr_data[i] = i+1
                cluster_Nr_data[j] = i+1 
                # It is important that the last line is not part of the "if j == 0" statement because the "if (cluster_Nr_data[i] == 0)" is also entered for values of j != 0 when
                # the center i has not yet been set to its own value because this thread was faster than the j = 0 thread.
                
        
    
    # contrary to Thomas cluster_Nr_data I do not store the number of the cluster to which a loc had been assigned in the form of 1 for the first found cluster center, 2 for the secondly found cluster.... 
    # Instead the cluster number is the position in the x_coords array of the coordinate which is the local maxima of this cluster enlarged by one 
    # (If a cluster contains several local maxima # the number will be the position of the local maxima that occurs earlier in x_coords). 
    # I use the position + 1 instead of the position to be sure that no cluster has the number 0,
    # as later all coordinates that do not belong to a true cluster will be assigend to the "cluster" 0.
    

    
    # start the kernel for every cluster center found:
    
    #d_assignment_radius = cuda.to_device(assignment_radius)  # for some reason the statement if current_distance <= assignment_radius does not work if we use the device assignment radius
    #s=timer()
    d_cluster_Nr_data = cuda.to_device(cluster_Nr_data)
    #e=timer()
    #print("to device cluster nr data (32)", e-s)
    
    
    
    blockdim_1d = 32
    griddim_1d = len(x_coords)//blockdim_1d + 1
    
    
    s1=timer()
    
    for i in range(0, len(x_coords)):
        
        if (local_maxima_data[i] == 1):
            
            assign_to_cluster_kernel[griddim_1d, blockdim_1d](i, assignment_radius, d_cluster_Nr_data, d_x_coords, d_y_coords)
            cuda.synchronize()
    
    e1 = timer()
    #print("cluster assignment", e1-s1)
    #print()
    
    #s=timer()      
    cluster_Nr_data = d_cluster_Nr_data.copy_to_host()    # necessary to copy it to host to do np.bincount     
    #e=timer()
    #print("to host cluster nr data", e-s)        
    
    
    # using this instead of the locs_in_cluster in the original code, we save using the time consuming np.unique on this vector
    # because the only thing that was used of locs_in_cluster was the number of locs in each cluster.
    nr_of_locs_in_cluster = np.bincount(cluster_Nr_data)
    

            
    '''true cluster or untrue cluster??  - via amount in clusters/localizations
    
    '''
    
    d_nr_of_locs_in_cluster = cuda.to_device(nr_of_locs_in_cluster)
    
    @cuda.jit
    def min_size_check_cluster_kernel(nr_of_locs_in_cluster, cluster_Nr_data, x_coords, y_coords, cluster_size_threshold):
        i = cuda.grid(1)
        
      
        if (i >= len(x_coords)):
            return
        
        if nr_of_locs_in_cluster[cluster_Nr_data[i]] <= cluster_size_threshold: 
            cluster_Nr_data[i] = 0          # coordinates in a too "small" cluster get reassigned to "cluster" 0 which will probably be deleted
    
    
    
    min_size_check_cluster_kernel[griddim_1d, blockdim_1d](d_nr_of_locs_in_cluster, d_cluster_Nr_data, d_x_coords, d_y_coords, cluster_size_threshold)
    
    
    
    '''
    =============================================================================
    2) Define clusters with the amount of neighbors (assign each localization to a cluster, #neighbors 2xNeNa & 1 NeNa)
    =============================================================================
    '''
    
    """
        This section was restructured. Code to calculate cluster properties and code to check if a cluster is a true cluster
        were arranged differently to implement the kernels more efficient.
    """
    
    
    """
    first rename the clusters and write their new indentification into cluster_Nr_data
    """
    
    cluster_Nr_data = d_cluster_Nr_data.copy_to_host() # necessary to calculate unique vector
    unique_clusters = np.unique(cluster_Nr_data)      
    amount_of_clusters = len(unique_clusters)
    
    
    s1=timer()
    
    
    com_x_cluster = np.zeros(amount_of_clusters, dtype=np.float32)            # com = center of mass
    com_y_cluster = np.zeros(amount_of_clusters, dtype=np.float32)
    
    elements_per_cluster = np.zeros(amount_of_clusters, dtype=np.int32)
    
    
    mean_frame_cluster = np.zeros(amount_of_clusters, dtype=np.float32)
    
    
    
    
    @cuda.jit
    def cluster_Nr_data_rename_kernel(cluster_Nr_data, unique_clusters):
    
        i = cuda.grid(1)
        
        if (i >= len(cluster_Nr_data)):
            return
        
        for j in range(len(unique_clusters)):
            if cluster_Nr_data[i] == unique_clusters[j]:
                cluster_Nr_data[i] = j 
        
        
    d_unique_clusters = cuda.to_device(unique_clusters)
    
    
    blockdim_1d = 32
    griddim_1d = len(cluster_Nr_data)//blockdim_1d + 1
    
    
    
    cluster_Nr_data_rename_kernel[griddim_1d, blockdim_1d](d_cluster_Nr_data, d_unique_clusters)
    
    
    
    
    '''==================
    Calculation of all important parameters for a cluster (center of mass...mean frame...)
    =====================''' 

    
    # frame number / 20
    window_search = frame[len(x_coords)-1]/20 #window search of appearing events in a certain timewindow (5% window), if it exceeds 40 % of total events occured, then cluster is untrue
    # this is the number of frames that form 5% of the total imaging time
    
    #calculate first the number of locs per window
    
    occuring_locs_in_window = np.zeros((amount_of_clusters,21), dtype=np.int32) # 21 windows: The 21st will only include localizations that occur in the very last frame. 
                                                                # The other 20 windows contain localizations occuring in the respective frame window
    
    
    @cuda.jit
    def cluster_props_kernel(cluster_Nr_data, unique_clusters, com_x_cluster, com_y_cluster, x_coords, y_coords, frame, occuring_locs_in_window, window_search, elements_per_cluster, mean_frame_cluster):
    
        j = cuda.grid(1)
        
        if (j >= len(unique_clusters)):
            return
        
        for i in range(len(x_coords)):
            if (j == cluster_Nr_data[i]):
                 
                com_x_cluster[j] += x_coords[i]
                com_y_cluster[j] += y_coords[i]
                elements_per_cluster[j] += 1
                
                mean_frame_cluster[j] += frame[i]
                #occuring_locs_in_window[j][frame[i]//window_search] += 1
                occuring_locs_in_window[j][int(frame[i]/window_search)] += 1
                #k = np.int(frame[i]/window_search)
                #occuring_locs_in_window[j][k] += 1
    
    
    start_t = timer()
    d_com_x_cluster = cuda.to_device(com_x_cluster)
    d_com_y_cluster = cuda.to_device(com_y_cluster)
    end_t = timer()
    #print("transfer to device com", end_t-start_t)
    d_frame = cuda.to_device(frame)
    d_occuring_locs_in_window  = cuda.to_device(occuring_locs_in_window)
    #d_window_search  = cuda.to_device(window_search)
    d_elements_per_cluster  = cuda.to_device(elements_per_cluster)
    d_mean_frame_cluster  = cuda.to_device(mean_frame_cluster)
    
    
    blockdim_1d = 32
    griddim_1d = len(unique_clusters)//blockdim_1d + 1
    
    
    #print(type(unique_clusters), type(com_x_cluster), type(occuring_locs_in_window), type(window_search))
    #print(type(unique_clusters[1]), type(com_x_cluster[1]), type(occuring_locs_in_window[1]), type(window_search))
    
    cluster_props_kernel[griddim_1d, blockdim_1d](d_cluster_Nr_data, d_unique_clusters, d_com_x_cluster, d_com_y_cluster, d_x_coords, d_y_coords, d_frame, d_occuring_locs_in_window, window_search, d_elements_per_cluster, d_mean_frame_cluster)
    
    
    #all calculations on the 0th cluster are unnecessary, right??? Stop it!
    
    
    
    e1 = timer()
    
    #print("new", e1-s1)
    

    
    s=timer()
    
    percentage_of_locs = np.zeros(amount_of_clusters)
    
    @cuda.jit
    def cluster_props_kernel2(com_x_cluster, com_y_cluster, occuring_locs_in_window, elements_per_cluster, mean_frame_cluster, percentage_of_locs):
    
        i = cuda.grid(1)
        
        if (i >= len(com_x_cluster)):
            return
        
        com_x_cluster[i] = com_x_cluster[i]/elements_per_cluster[i]
        com_y_cluster[i] = com_y_cluster[i]/elements_per_cluster[i]
        mean_frame_cluster[i] = mean_frame_cluster[i]/elements_per_cluster[i]  
        
        percentage_of_locs[i] = 0
        for j in range(0,21):           
            new_percentage = occuring_locs_in_window[i][j]/elements_per_cluster[i]
            if new_percentage > percentage_of_locs[i]:
                percentage_of_locs[i] = new_percentage
            
    
    
    d_percentage_of_locs = cuda.to_device(percentage_of_locs)
    
    blockdim_1d = 32
    griddim_1d = len(unique_clusters)//blockdim_1d + 1
    
    
    cluster_props_kernel2[griddim_1d, blockdim_1d](d_com_x_cluster, d_com_y_cluster, d_occuring_locs_in_window, d_elements_per_cluster, d_mean_frame_cluster, d_percentage_of_locs)
    
    
    
    
    """
    4) Check clusters for beeing true or false
    a) Repetitive visits over the course of imaging? 
    ->(via cumulative distribution cutoffs -> if there is a jump in the cumulative distribution, kick it out)
    """
    
    true_cluster = np.zeros(amount_of_clusters,dtype=np.int8)
    
    
    @cuda.jit
    def true_cluster_kernel(mean_frame_cluster, percentage_of_locs, true_cluster, frame_nr):
        #def true_cluster(mean_frame_cluster, d_percentage_of_locs,  frame_nr):
    
        i = cuda.grid(1)
        
        if (i >= len(mean_frame_cluster)):
            return
        
        if (percentage_of_locs[i] < 0.8) and (mean_frame_cluster[i] < frame_nr*0.8) and (mean_frame_cluster[i] > frame_nr*0.2): # and (empty_window_count[i] < 3):
            true_cluster[i] = 1
        if (percentage_of_locs[i] >= 0.8): # or (empty_window_count[i] >= 3):
            true_cluster[i] = 0
           
    
    
    d_true_cluster = cuda.to_device(true_cluster)
    
    blockdim_1d = 32
    griddim_1d = len(unique_clusters)//blockdim_1d + 1
    
    
    frame_nr = frame[len(x_coords)-1]
    
    
    true_cluster_kernel[griddim_1d, blockdim_1d](d_mean_frame_cluster, d_percentage_of_locs, d_true_cluster, frame_nr)
    
    
    
    e=timer()
    
    #print("new cluster props:", e-s)
    
    
    
    
    '''========================
    7) write new dataset with only true spots
    -> create new data 2 dataset
    ============================'''
    
    
    '''
    LOCS_DTYPE = [("frame"),("x"),("y"),("z"),("photons"),("sx"),("sy"),("bg"),("lpx"),("lpy")]
    '''
    s = timer()
    
    data2_frames = []
    data2_x = []
    data2_y = []
    data2_photons = []
    data2_sx = []
    data2_sy = []
    data2_bg = []
    data2_lpx = []
    data2_lpy = []
    data2_ellipticity = []
    data2_net_gradient = []
    data2_group = []
    
    cluster_Nr_data = d_cluster_Nr_data.copy_to_host()    # necessary to copy it to host to do np.bincount     
    com_x_cluster = d_com_x_cluster.copy_to_host()        
    com_y_cluster= d_com_y_cluster.copy_to_host()        
    mean_frame_cluster = d_mean_frame_cluster.copy_to_host()  
    true_cluster = d_true_cluster.copy_to_host()  
    
    frames = data['frame']
    photons = data['photons']
    sx = data['sx']
    sy = data['sy']
    bg = data['bg']
    lpx = data['lpx']
    lpy = data['lpy']
    ellipticity = data['ellipticity']
    net_gradient = data['net_gradient']
    
    # Das müsste auch smarter gehen
    for i in range(0, len(x_coords)):
        if (cluster_Nr_data[i] != 0) and (true_cluster[cluster_Nr_data[i]] == 1):
    
        #if (true_cluster[cluster_Nr_data[i]] == 1):
            data2_frames.append(frames[i])
            data2_x.append(x_coords[i])
            data2_y.append(y_coords[i])
            data2_photons.append(photons[i])
            data2_sx.append(sx[i])
            data2_sy.append(sy[i])
            data2_bg.append(bg[i])
            data2_lpx.append(lpx[i])
            data2_lpy.append(lpy[i])
            data2_ellipticity.append(ellipticity[i])
            data2_net_gradient.append(net_gradient[i])
            data2_group.append(cluster_Nr_data[i])
    
    m1 = timer()
    #write true clusters into file
    
    amount_of_true_clusters = 0
    
    for i in range(0, amount_of_clusters):
        if true_cluster[i]==1:
            amount_of_true_clusters = amount_of_true_clusters+1
    
    m2 = timer()
    
        
    new_com_x_cluster = np.zeros(amount_of_true_clusters)
    new_com_y_cluster = np.zeros(amount_of_true_clusters)
    
    new_mean_frame_cluster = np.zeros(amount_of_true_clusters)
    
    
    counter=0
    
    for i in range(0, amount_of_clusters):
        if true_cluster[i]==1:
            new_com_x_cluster[counter] = com_x_cluster[i]
            new_com_y_cluster[counter] = com_y_cluster[i]
            new_mean_frame_cluster[counter] = mean_frame_cluster[i]
            counter=counter+1
    
    
    e=timer()
    #print("data2:", m1-s)
    #print("amount of clusters", m2-m1)
    #print("new_com_x_cluster:", e-m2)
    
    
    
    
    
    
    s = timer()
    #np.savez('%s_varsD%d_%d' %(filename, threshold_radius, cluster_size_threshold), higher_neighbors_data2=higher_neighbors_data2,data2_x=data2_x,data2_y=data2_y, data2_frames=data2_frames, data2_group=data2_group, new_com_x_cluster=new_com_x_cluster, new_com_y_cluster=new_com_y_cluster,amountOfNeighbors_data=amountOfNeighbors_data, x_coords=x_coords, y_coords=y_coords) 
    
    
    '''
    Generating hdf5 file for picasso render
    '''
    #filename = filename[0:len(filename)-5]
    
    
    import h5py as _h5py
    import pandas as pd
    import numpy as _np
    
    data_cl = {'frame': data2_frames, 'x': data2_x, 'y': data2_y, 'photons': data2_photons, 'sx': data2_sx, 'sy': data2_sy, 'bg': data2_bg, 'lpx': data2_lpx,'lpy': data2_lpy, 'ellipticity': data2_ellipticity, 'net_gradient': data2_net_gradient, 'group': data2_group}
    
    
    df = pd.DataFrame(data_cl, index=range(len(data2_x)))
    df = df.astype({'frame': 'u4', 'x': 'f4', 'y': 'f4', 'photons': 'f4', 'sx': 'f4', 'sy': 'f4', 'bg': 'f4', 'lpx': 'f4','lpy': 'f4', 'ellipticity': 'f4', 'net_gradient': 'f4', 'group': 'u4'})

    df2 = df.reindex(columns = ['frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy', 'ellipticity', 'net_gradient', 'group'], fill_value=1)
    
    path = os.path.split(filename)[0] + "/"
    filename_old = os.path.split(filename)[1]
    filename_new = '%s_ClusterD%s_%d.hdf5' % (filename_old[:-5], threshold_radius_str, cluster_size_threshold)
    tools.picasso_hdf5(df2, filename_new, filename_old, path)
    """
    LOCS_DTYPE = [
    ('frame', 'u4'),
    ('x', 'f4'),
    ('y', 'f4'),
    ('group', 'u4'),
    ('photons', 'f4'),
    ('sx', 'f4'),
    ('sy', 'f4'),
    ('bg', 'f4'),
    ('lpx', 'f4'),
    ('lpy', 'f4')
    """
    e = timer()
    #print("save:", e-s) 
    
    
    
    
    '''
    Generating hdf5 file for picasso render with the weighted centers as loclaizaitons
    '''
    grouped = df2.groupby("group")
    
    # Test if every cluster is a Gaussian (Shapiro Wilk test)
    from scipy.stats import shapiro
    shapiro_df = (grouped.apply(lambda x: pd.Series(shapiro(x), index=['W', 'P'])).reset_index())
    #print(shapiro_df)
    
    
    x_av_wtd = grouped.apply(lambda x: np.average(x['x'],weights=1/x['lpx']/x['lpx']))
    x_av_wtd.name = "x_av_wtd"
    #print(x_av_wtd.name)
    y_av_wtd  = grouped.apply(lambda x: np.average(x['y'],weights=1/x['lpy']/x['lpy']))
    y_av_wtd.name = "y_av_wtd"
    
    #xx_av_wtd = grouped.apply(lambda x: np.average(x['x']*x['x'],weights=1/x['lpx']/x['lpx']))
    #xx_av_wtd.name = "xx_av_wtd"
    group_size = grouped.size()
    group_size.name = "group_size"
    """
    print(x_av_wtd)
    print(type(x_av_wtd))
    #print(xx_av_wtd)
    print(group_size)
    print(type(group_size))
    """
    
    #x = x_av_wtd.to_frame()
    #print(x.to_string())
    #print(x)
    #print(type(x))
    '''
    merged_group_info = pd.merge(x_av_wtd, xx_av_wtd, on='group')
    merged_group_info2 = merged_group_info.merge(group_size, on = 'group')
    #merged_group_info = x_av_wtd.to_frame().merge(xx_av_wtd, on='group')
    #merged_group_info2 = merged_group_info.merge(group_size, on = 'group')
    print(merged_group_info2)
    print(type(merged_group_info2))
    print(merged_group_info2.shape)
    print(list(merged_group_info2.columns))
    
    '''
    ''' # Note: This calculation leads to a cancellation error!!!! F.ex. negative variances are produced.
    def std_wtd2(group, x_av_wtd, xx_av_wtd, group_size):
        x_av = group[x_av_wtd]
        xx_av = group[xx_av_wtd]
        gr_size = group[group_size]
        
        return (xx_av-x_av*x_av)*gr_size/(gr_size-1)
    
    #test = merged_group_info2.apply(lambda x: (x['xx_av_wtd']-x['x_av_wtd']*x['x_av_wtd'])*x['group_size']/(x['group_size']-1))
    test2 = std_wtd2(merged_group_info2, "x_av_wtd", "xx_av_wtd", "group_size")
    print(test2)
    
    
    '''
    def error_sums_wtd(group, x, lpx):
        x = group[x]
        w = 1/group[lpx]**2
        #gr_size = group.size()
        #x_av = (w * x).sum() / w.sum()
        #return (w * (x - (w * x).sum() / w.sum())**2).sum() / w.sum() * gr_size/(gr_size-1)
        return (w * (x - (w * x).sum() / w.sum())**2).sum() / w.sum()
    
    
    
    #var_x = grouped.apply(std_wtd, "x", "lpx")*group_size/(group_size - 1)
    sed_x = np.sqrt(grouped.apply(error_sums_wtd, "x", "lpx")/(group_size - 1))
    sed_y = np.sqrt(grouped.apply(error_sums_wtd, "y", "lpy")/(group_size - 1))
    sed_xy = np.mean([sed_x, sed_y], axis = 0)
    
    #print(sed_x)
    
    '''
    # Function to calculate the Gaussian with constants a, b, and c
    def gaussian(x, a, b, c):
        return a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))
    
    
    # Fit the dummy Gaussian data
    pars, cov = curve_fit(f=gaussian, xdata=test.x, ydata=test.y, p0=[0, 0, 0], bounds=(-np.inf, np.inf))
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    
    '''
    
    
    
    
    
    group_means = grouped.mean()
    #print(group_means)
    #print(group_means['frame'].values.tolist())
    data3_frames = group_means['frame'].values.tolist()
    data3_x = x_av_wtd.values.tolist()
    data3_y = y_av_wtd.values.tolist()
    data3_photons = group_means['photons'].values.tolist()
    data3_sx = group_means['sx'].values.tolist()
    data3_sy = group_means['sy'].values.tolist()
    data3_bg = group_means['bg'].values.tolist()
    data3_lpx = sed_xy.tolist()
    data3_lpy = sed_xy.tolist()
    print(data['group'])
    print(np.mean(np.array(data['group'])))
    data3_group = np.full(shape = len(data3_lpy), fill_value = np.mean(np.array(data['group']))) # group of origami, not of clusters in origami
    data3_n = group_size.values.tolist()
    
    '''
    Generating hdf5 file for picasso render
    '''
    import h5py as _h5py
    import pandas as pd
    import numpy as _np
    
    data = {'frame': data3_frames, 'x': data3_x, 'y': data3_y, 'photons': data3_photons, 'sx': data3_sx, 'sy': data3_sy, 'bg': data3_bg, 'lpx': data3_lpx,'lpy': data3_lpy, 'group': data3_group, 'n': data3_n}
    
    
    
    df = pd.DataFrame(data, index=range(len(data3_x)))
    df = df.astype({'frame': 'u4', 'x': 'f4', 'y': 'f4', 'photons': 'f4', 'sx': 'f4', 'sy': 'f4', 'bg': 'f4', 'lpx': 'f4','lpy': 'f4', 'group': 'u4'})
    df3 = df.reindex(columns = ['frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy', 'group', 'n'], fill_value=1)
    
    try:
        os.mkdir(os.path.split(filename)[0] + '/AdditionalOutputs')
        
    except OSError:
        print ("transf_overview folder already exists")


    path = os.path.split(filename)[0] + "/"
    filename_old = os.path.split(filename)[1]
    filename_new = '%s_resi_%s_%d.hdf5' % (filename_old[:-5], threshold_radius_str, cluster_size_threshold)
    tools.picasso_hdf5(df3, filename_new, filename_old, path)


    '''
    Save Shapiro results to csv file
    '''

    path = os.path.split(filename)[0] + "/AdditionalOutputs/"
    shapiro_filename = path + os.path.split(filename)[1][:-5]
    shapiro_df.to_csv('%s_shapiro-wilk_%s_%d.csv' %(shapiro_filename, threshold_radius_str, cluster_size_threshold), index=False)
    
    
    
    '''
    Save npz with arrays used for postprocessing
    '''
    np.savez('%s_varsD%s_%d' %(filename[:-5], threshold_radius_str, cluster_size_threshold),data2_x=data2_x,data2_y=data2_y, data2_frames=data2_frames, data2_group=data2_group, new_com_x_cluster=data3_x, new_com_y_cluster=data3_y,amountOfNeighbors_data=amountOfNeighbors_data, x_coords=x_coords, y_coords=y_coords) 
    # the array is still named ...com... for center of mass. however, now the npz file now contains the weighted centers of the clusters.
    
    
    
    
    
    end_all = timer()
    #print("total runtime:", end_all - start_all)
