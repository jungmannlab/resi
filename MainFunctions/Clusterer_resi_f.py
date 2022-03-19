
from timeit import default_timer as timer
start_all = timer()
from numba import cuda


import os
import h5py
import numpy as np
import sys
import math
import tools



def clusterer_resi(hdf5_file, radius, min_cluster_size, radius_z=0):

    #Read in HDF5-File
    filename = hdf5_file
    f1 = h5py.File(filename, 'r')
    a_group_key = list(f1.keys())[0]
    data = np.array(f1[a_group_key])
    
    threshold_radius = float(radius)
    threshold_radius_str = str(radius)

    cluster_size_threshold = min_cluster_size

    pl = 130 # pixel length [nm]
    x_coords = np.ascontiguousarray(data['x'])
    y_coords = np.ascontiguousarray(data['y'])
    frame = np.ascontiguousarray(data['frame'])

    print()
    print("number of coordinates:", len(x_coords))

    
    try:
        z_coords =  np.ascontiguousarray(data['z'])/pl
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
            #radius_z = float(float(sys.argv[4]))
            z_coords = np.ascontiguousarray(data['z'])/pl
            
    
    
    
    # A dummy variable for numba's 'optional' arguments
    dum_arr = np.zeros(1)
    dum_rad = 100000 #just some large value for the threshold radius in z direction  
                  #that is unlikely large
    

    
    
    '''
    =============================================================================
    1) Calculate amount of neighbors within the threshold radius around every localization 
    =============================================================================
    '''
    '''
    Now the amount of neighbors is calculated inside a cuda kernel, where each 
    thread corresponds to one localization. Inside the kernel, each thread loops 
    again over all localizations to find the localizations that are within the 
    radius around the localization centered in this thread.
    '''
    
    #check for each cluster how many neighbors it has within this distance
    amountOfNeighbors_data = np.zeros(len(x_coords), dtype=np.int16)
    d_amountOfNeighbors_data = cuda.to_device(amountOfNeighbors_data)
    # datatype can be even less consuming as it only takes values 0 and 1
    
    
    # transfer the coordinates to the devide (GPU) only once, as they will be used 
    # by several kernels.
    d_x_coords = cuda.to_device(x_coords)
    d_y_coords = cuda.to_device(y_coords)
    if flag_3D:
        d_z_coords = cuda.to_device(z_coords)
        
    
    # calculates for each coordinate the amount of neighbours within the entered 
    # radius
    @cuda.jit
    def amountOfNeighbors_kernel_new(amountOfNeighbors_data, threshold_radius, 
                                     x_coords, y_coords, z_coords, radius_z):
        # the 3D implementation with z_coords is weird because numba does not seem
        # to like optional arguments, therefore, I needed to make if a bit complicated
        # Note that when calling the kernels in 2D, z_coords is set to np.zeros(1)
        i = cuda.grid(1) # i = absolute thread position
        if i >= len(x_coords):
            return
        sumtemp = 0
        for j in range(len(x_coords)):
            if z_coords.size == 1:
                distance = ((x_coords[i]-x_coords[j])**2+\
                           (y_coords[i]-y_coords[j])**2)*(pl**2)
            else:
                distance = ((x_coords[i]-x_coords[j])**2+\
                            (y_coords[i]-y_coords[j])**2+\
                            ((threshold_radius/radius_z)*\
                            (z_coords[i]-z_coords[j]))**2)*(pl**2)
            
            if distance <= threshold_radius**2:
                sumtemp += 1  
        amountOfNeighbors_data[i] = sumtemp - 1
        # the -1 replaces the condition if i!=j. Testing this vor every element 
        # in x_coords costs time and the result what happens if i = j is always 0. 
        # so we can just reduce the amount of neighbors by 1 to get the real 
        # amount of neighbors without counting the distance of i to itself.
         
                    
    # call the kernel
    blockdim_1d = 32
    griddim_1d = len(x_coords)//blockdim_1d + 1
    
    if not flag_3D:
        amountOfNeighbors_kernel_new[griddim_1d, blockdim_1d](d_amountOfNeighbors_data,
                                              threshold_radius, d_x_coords, d_y_coords,
                                              dum_arr, dum_rad)
    else:
        amountOfNeighbors_kernel_new[griddim_1d, blockdim_1d](d_amountOfNeighbors_data,
                                              threshold_radius, d_x_coords, d_y_coords, 
                                              d_z_coords, radius_z)
    cuda.synchronize()
    

    
    
    
    '''
    =============================================================================
    1b) find local maxima in distance r
    =============================================================================
    '''
    '''
    Now the amount of neighbors is calculated inside a cuda kernel, where each 
    thread corresponds to one localization. Inside the kernel, each thread loops 
    again over all localizations to find out whether there is another localizations
    within the radius that has more neighbors than the localization centered in 
    this thread.
    
    '''
    
    # description how it works:
    # compare the amount of neighbors within the threshold_radius of a coordinate 
    # with the respective amount of all other coordinates in the same box that are 
    # closer than the threshold distance. E.g. compare i to j: i has more neighbors
    # than j -> set local_maxima_data[i] to 1 then compare i to k: i has less 
    # neighbors than k: -> set local_maxima_data[i] to 0 and exit the second loop 
    # and start with a new coordinate to compare to all the other ones in its box
    # if a coordinate gets assigned a 0, it is definetly not the local maximum in 
    # the box this should end with one coordinate assigned to 1 and all others to 
    # 0 (only exception: some coordinates within the same box have exactly the same
    # number of neighbors)
    
    local_maxima_data = np.zeros(len(x_coords), dtype=np.int8)
    d_local_maxima_data = cuda.to_device(local_maxima_data)

    
    @cuda.jit
    def local_maxima_kernel(local_maxima_data, amountOfNeighbors_data,
                            threshold_radius, x_coords, y_coords, z_coords, radius_z):
        
        i = cuda.grid(1) # i = absolute thread position
        if (i >= len(x_coords)):
            return
        
        for j in range(len(x_coords)):
            if z_coords.size == 1:
                distance = ((x_coords[i]-x_coords[j])**2+\
                            (y_coords[i]-y_coords[j])**2)*(pl**2) 
            else:
                distance = ((x_coords[i]-x_coords[j])**2+\
                            (y_coords[i]-y_coords[j])**2+\
                            ((threshold_radius/radius_z)*\
                            (z_coords[i]-z_coords[j]))**2)*(pl**2) 
            
            if (distance <= threshold_radius**2) and (amountOfNeighbors_data[i] >= amountOfNeighbors_data[j]):
                local_maxima_data[i] = 1
    
            if (distance <= threshold_radius**2) and (amountOfNeighbors_data[i] < amountOfNeighbors_data[j]):
                local_maxima_data[i] = 0
                break
                # contrary to the non GPU code the break statement does not speed up 
                # the code significantly (probably due to the organizations in warps). 
                # however it is important to have it because then we make sure that 
                # local_maxima_data[i] really is 0 if it occurs once. Without the break
                # the for loop continues and the 0 can be changed to 1 in the next 
                # round. And if by chance in no further round it is set to 0 we will 
                # get a wrong result.
    
    # Within the given radius only one cluster center can exist (with value local_maxima_data[i]=1. 
    # However, if there are two locs with exactly the same amount of neighbors they 
    # will both have the value 1
    

                    
    # call the kernel
    blockdim_1d = 32
    griddim_1d = len(x_coords)//blockdim_1d + 1
    
    if not flag_3D:
        local_maxima_kernel[griddim_1d, blockdim_1d](d_local_maxima_data, d_amountOfNeighbors_data,
                                                 threshold_radius, d_x_coords,
                                                 d_y_coords, dum_arr, dum_rad)
    else:
        local_maxima_kernel[griddim_1d, blockdim_1d](d_local_maxima_data, d_amountOfNeighbors_data,
                                                 threshold_radius, d_x_coords, 
                                                 d_y_coords, d_z_coords, radius_z)
    cuda.synchronize()

    local_maxima_data = d_local_maxima_data.copy_to_host()

    
    '''
    
    Define clusters according to maxima.
    
    Note that the following kernel also changes how the clusters are named and thus
    how the content of in cluster_Nr_data will look like compared to the CPU 
    version. This simplified the implementation of the kernel. In the kernel 
    cluster_Nr_data_rename_kernel the numbering of clusters will be translated to 
    how it is in the CPU clusterer.
    
    '''
    
    # for each localization it will contain the information to which cluster it 
    # belongs. If the coordinate at index j is in the cluster of the localization 
    # i, which is the cluster center of the cluster, then the value of 
    # cluster_Nr_data[j] will be i+1
    
    
    cluster_Nr_data = np.zeros(len(x_coords), dtype=np.int32)

    @cuda.jit
    def assign_to_cluster_kernel(i, threshold_radius, cluster_Nr_data, 
                                 x_coords, y_coords, z_coords, radius_z):
    
        j = cuda.grid(1)
        if (j >= len(x_coords)):
            return    
        
        if z_coords.size == 1:
            distance = ((x_coords[i]-x_coords[j])**2+\
                       (y_coords[i]-y_coords[j])**2)*(pl**2)
        else:
            distance = ((x_coords[i]-x_coords[j])**2+\
                        (y_coords[i]-y_coords[j])**2+\
                        ((threshold_radius/radius_z)*\
                        (z_coords[i]-z_coords[j]))**2)*(pl**2)
    
        if distance <= threshold_radius**2:
    
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

    d_cluster_Nr_data = cuda.to_device(cluster_Nr_data)
    
    blockdim_1d = 32
    griddim_1d = len(x_coords)//blockdim_1d + 1
    
    for i in range(len(x_coords)):
        
        if local_maxima_data[i] == 1: 
            if not flag_3D:
                assign_to_cluster_kernel[griddim_1d, blockdim_1d](i, threshold_radius,
                                        d_cluster_Nr_data, d_x_coords,
                                        d_y_coords, dum_arr, dum_rad)
                cuda.synchronize()
            else:
                assign_to_cluster_kernel[griddim_1d, blockdim_1d](i, threshold_radius,
                                        d_cluster_Nr_data, d_x_coords, 
                                        d_y_coords, d_z_coords, radius_z)
                cuda.synchronize()
    
     
    cluster_Nr_data = d_cluster_Nr_data.copy_to_host()    
    # necessary to copy it to host to do np.bincount     

    # using this instead of the locs_in_cluster in the original code, we save using the time consuming np.unique on this vector
    # because the only thing that was used of locs_in_cluster was the number of locs in each cluster.
    nr_of_locs_in_cluster = np.bincount(cluster_Nr_data)
    

            
    '''
    true cluster or untrue cluster??  - via amount in clusters/localizations
    '''
    
    d_nr_of_locs_in_cluster = cuda.to_device(nr_of_locs_in_cluster)
    
    @cuda.jit 
    def min_size_check_cluster_kernel(nr_of_locs_in_cluster, cluster_Nr_data, 
                                      x_coords, cluster_size_threshold):
    
        i = cuda.grid(1) 
        if (i >= len(x_coords)):
            return
    
        if nr_of_locs_in_cluster[cluster_Nr_data[i]] <= cluster_size_threshold: 
            cluster_Nr_data[i] = 0 # coordinates in a too "small" cluster get 
            #reassigned to "cluster" 0 which will probably be deleted
    
    min_size_check_cluster_kernel[griddim_1d, blockdim_1d](d_nr_of_locs_in_cluster,
                              d_cluster_Nr_data, d_x_coords, cluster_size_threshold)
    cuda.synchronize() 
    

    '''
    =============================================================================
    2) Define clusters with the amount of neighbors (assign each localization to a 
       cluster, #neighbors 2xNeNa & 1 NeNa)
    =============================================================================
    '''
    
    """
    This section was restructured. Code to calculate cluster properties and code to
    check if a cluster is a true cluster were arranged differently to implement 
    the kernels more efficiently.
    
    First rename the clusters and write their new indentification into cluster_Nr_data
    """
    
    cluster_Nr_data = d_cluster_Nr_data.copy_to_host() # necessary to calculate unique vector
    unique_clusters = np.unique(cluster_Nr_data)      
    amount_of_clusters = len(unique_clusters)

    com_x_cluster = np.zeros(amount_of_clusters, dtype=np.float32)            # com = center of mass
    com_y_cluster = np.zeros(amount_of_clusters, dtype=np.float32)
    if flag_3D:
        com_z_cluster = np.zeros(amount_of_clusters, dtype=np.float32)
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
    
    cluster_Nr_data_rename_kernel[griddim_1d, blockdim_1d](d_cluster_Nr_data,
                                                           d_unique_clusters) 
    cuda.synchronize()
    
    
    
    '''
    ===========================================================================
    Calculation of all important parameters for a cluster (center of mass...mean 
    frame...)
    ===========================================================================
    ''' 
    
    # frame number / 20
    window_search = frame[len(x_coords)-1]/20 #window search of appearing events in
    # a certain timewindow (5% window), if it exceeds 40 % of total events occured,
    # then cluster is untrue
    
    #calculate first the number of locs per window
    occuring_locs_in_window = np.zeros((amount_of_clusters,21), dtype=np.int32) 
    # 21 windows: The 21st will only include localizations that occur in the very last frame. 
    # The other 20 windows contain localizations occuring in the respective frame window
    @cuda.jit
    def cluster_props_kernel(cluster_Nr_data, unique_clusters, com_x_cluster, 
                             com_y_cluster, x_coords, y_coords, frame, 
                             occuring_locs_in_window, window_search, 
                             elements_per_cluster, mean_frame_cluster, 
                             z_coords, com_z_cluster):
    
        j = cuda.grid(1)    
        if (j >= len(unique_clusters)):
            return
        
        for i in range(len(x_coords)):
            if (j == cluster_Nr_data[i]):
                
                com_x_cluster[j] += x_coords[i]
                com_y_cluster[j] += y_coords[i]
                if z_coords.size != 1:
                    com_z_cluster[j] += z_coords[i]
                elements_per_cluster[j] += 1
                mean_frame_cluster[j] += frame[i]
                occuring_locs_in_window[j][int(frame[i]/window_search)] += 1
                

    d_com_x_cluster = cuda.to_device(com_x_cluster)
    d_com_y_cluster = cuda.to_device(com_y_cluster)
    if flag_3D:
        d_com_z_cluster = cuda.to_device(com_z_cluster)
    d_frame = cuda.to_device(frame)
    d_occuring_locs_in_window  = cuda.to_device(occuring_locs_in_window)
    d_elements_per_cluster  = cuda.to_device(elements_per_cluster)
    d_mean_frame_cluster  = cuda.to_device(mean_frame_cluster)
     
    
    blockdim_1d = 32
    griddim_1d = len(unique_clusters)//blockdim_1d + 1
    
    if not flag_3D:
        cluster_props_kernel[griddim_1d, blockdim_1d](d_cluster_Nr_data,
                         d_unique_clusters, d_com_x_cluster, d_com_y_cluster,
                         d_x_coords, d_y_coords, d_frame, d_occuring_locs_in_window,
                         window_search, d_elements_per_cluster, d_mean_frame_cluster,
                         dum_arr, dum_arr)
    else:
         cluster_props_kernel[griddim_1d, blockdim_1d](d_cluster_Nr_data,
                         d_unique_clusters, d_com_x_cluster, d_com_y_cluster,
                         d_x_coords, d_y_coords, d_frame, d_occuring_locs_in_window,
                         window_search, d_elements_per_cluster, 
                         d_mean_frame_cluster, d_z_coords, d_com_z_cluster)
        
    cuda.synchronize()


    #all calculations on the 0th cluster are unnecessary, right??? Stop it!
   

    
    percentage_of_locs = np.zeros(amount_of_clusters)
    
    @cuda.jit
    def cluster_props_kernel2(com_x_cluster, com_y_cluster, occuring_locs_in_window,
                              elements_per_cluster, mean_frame_cluster,
                              percentage_of_locs, com_z_cluster):
    
        i = cuda.grid(1)    
        if (i >= len(com_x_cluster)):
            return
    
        com_x_cluster[i] /= elements_per_cluster[i]
        com_y_cluster[i] /= elements_per_cluster[i]
        if com_z_cluster.size != 1:
            com_z_cluster[i] /= elements_per_cluster[i]
        mean_frame_cluster[i] = mean_frame_cluster[i]/elements_per_cluster[i]  
        percentage_of_locs[i] = 0
        
        for j in range(21):           
            new_percentage = occuring_locs_in_window[i][j]/elements_per_cluster[i]
            if new_percentage > percentage_of_locs[i]:
                percentage_of_locs[i] = new_percentage
          
    d_percentage_of_locs = cuda.to_device(percentage_of_locs)
    
    blockdim_1d = 32
    griddim_1d = len(unique_clusters)//blockdim_1d + 1
    
    if not flag_3D:
        cluster_props_kernel2[griddim_1d, blockdim_1d](d_com_x_cluster, d_com_y_cluster,
                          d_occuring_locs_in_window, d_elements_per_cluster, 
                          d_mean_frame_cluster, d_percentage_of_locs, dum_arr)
    else:
        cluster_props_kernel2[griddim_1d, blockdim_1d](d_com_x_cluster, d_com_y_cluster,
                          d_occuring_locs_in_window, d_elements_per_cluster, 
                          d_mean_frame_cluster, d_percentage_of_locs, d_com_z_cluster)
    cuda.synchronize()
    
    
    
    
    """
    4) Check clusters for beeing true or false
    a) Repetitive visits over the course of imaging? 
    ->(via cumulative distribution cutoffs -> if there is a jump in the cumulative
       distribution, kick it out)
    
    """
    
    true_cluster = np.zeros(amount_of_clusters,dtype=np.int8)
    
    @cuda.jit
    def true_cluster_kernel(mean_frame_cluster, percentage_of_locs, true_cluster, frame_nr):
    
        i = cuda.grid(1)
        if (i >= len(mean_frame_cluster)):
            return    
    
        if (percentage_of_locs[i] < 0.8) and (mean_frame_cluster[i] < frame_nr*0.8) and (mean_frame_cluster[i] > frame_nr*0.2): # and (empty_window_count[i] < 3):
            true_cluster[i] = 1
    
        if (percentage_of_locs[i] >= 0.8):
            true_cluster[i] = 0
        
    d_true_cluster = cuda.to_device(true_cluster)
    
    blockdim_1d = 32
    griddim_1d = len(unique_clusters)//blockdim_1d + 1
    
    frame_nr = frame[len(x_coords)-1]
    
    true_cluster_kernel[griddim_1d, blockdim_1d](d_mean_frame_cluster, d_percentage_of_locs,
                                                 d_true_cluster, frame_nr)
    cuda.synchronize()
    

    
    '''
    ============================================================================
    5) write new dataset with only true spots -> create new data 2 dataset
    ============================================================================
    
    LOCS_DTYPE = [("frame"),("x"),("y"),("z"),("photons"),("sx"),("sy"),("bg"),
                  ("lpx"),("lpy")]
    '''
    
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
    data2_netgradient = []
    if flag_3D:
        data2_z = []
        data2_d_zcalib = []
    data2_group = []
    
    cluster_Nr_data = d_cluster_Nr_data.copy_to_host()   
    com_x_cluster = d_com_x_cluster.copy_to_host()        
    com_y_cluster= d_com_y_cluster.copy_to_host()   
    if flag_3D:
        com_z_cluster = d_com_z_cluster  .copy_to_host()   
    mean_frame_cluster = d_mean_frame_cluster.copy_to_host()  
    true_cluster = d_true_cluster.copy_to_host()  
    
    for i in range(len(x_coords)):
    
        if (cluster_Nr_data[i] != 0) and (true_cluster[cluster_Nr_data[i]] == 1):
    
            data2_frames.append(data['frame'][i])
            data2_x.append(x_coords[i])
            data2_y.append(y_coords[i])
            data2_photons.append(data['photons'][i])
            data2_sx.append(data['sx'][i])
            data2_sy.append(data['sy'][i])
            data2_bg.append(data['bg'][i])
            data2_lpx.append(data['lpx'][i])
            data2_lpy.append(data['lpy'][i])
            data2_ellipticity.append(data['ellipticity'][i])
            data2_netgradient.append(data['net_gradient'][i])
            if flag_3D: 
                data2_z.append(z_coords[i]*pl)
                data2_d_zcalib.append(data['d_zcalib'][i])
            data2_group.append(cluster_Nr_data[i])
    

    #write true clusters into file
    amount_of_true_clusters = np.sum(true_cluster)
        
    
    new_com_x_cluster = np.zeros(amount_of_true_clusters)
    new_com_y_cluster = np.zeros(amount_of_true_clusters)
    if flag_3D:
        new_com_z_cluster = np.zeros(amount_of_true_clusters)
    new_mean_frame_cluster = np.zeros(amount_of_true_clusters)
    counter=0
    
    for i in range(amount_of_clusters):
    
        if true_cluster[i] == 1:
            new_com_x_cluster[counter] = com_x_cluster[i]
            new_com_y_cluster[counter] = com_y_cluster[i]
            if flag_3D:
                new_com_z_cluster[counter] = com_z_cluster[i]
            new_mean_frame_cluster[counter] = mean_frame_cluster[i]
            counter += 1
        

    
    '''
    Generating hdf5 file for picasso render
    '''
    
    import h5py as _h5py
    import pandas as pd
    import numpy as _np
    
    if not flag_3D:
        data_cl = {'frame': data2_frames, 'x': data2_x, 'y': data2_y, 
                   'photons': data2_photons, 'sx': data2_sx, 'sy': data2_sy, 
                   'bg': data2_bg, 'lpx': data2_lpx,'lpy': data2_lpy, 
                   'ellipticity': data2_ellipticity, 'net_gradient': data2_netgradient, 
                   'group': data2_group}
        
        df = pd.DataFrame(data_cl, index=range(len(data2_x)))
        df = df.astype({'frame': 'u4', 'x': 'f4', 'y': 'f4', 'photons': 'f4', 
                        'sx': 'f4', 'sy': 'f4', 'bg': 'f4', 'lpx': 'f4','lpy': 'f4',
                        'ellipticity': 'f4', 'net_gradient': 'f4', 'group': 'u4'})
        df2 = df.reindex(columns = ['frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 
                                            'lpx', 'lpy', 'ellipticity', 'net_gradient', 'group'], fill_value=1)
            
        path = os.path.split(filename)[0] + "/"
        filename_old = os.path.split(filename)[1]
        filename_new = '%s_ClusterD%s_%d.hdf5' % (filename_old[:-5], threshold_radius_str, cluster_size_threshold)
        tools.picasso_hdf5(df2, filename_new, filename_old, path)

    else:
        data_cl = {'frame': data2_frames, 'x': data2_x, 'y': data2_y, 'z': data2_z,
                   'photons': data2_photons, 'sx': data2_sx, 
                   'sy': data2_sy, 'bg': data2_bg, 'lpx': data2_lpx, 'lpy': data2_lpy,
                   'ellipticity': data2_ellipticity, 'net_gradient': data2_netgradient,
                   'd_zcalib': data2_d_zcalib, 'group': data2_group}
            
        df = pd.DataFrame(data_cl, index=range(len(data2_x)))
        df = df.astype({'frame': 'u4', 'x': 'f4', 'y': 'f4', 'z': 'f4', 'photons': 'f4',
                        'sx': 'f4', 'sy': 'f4', 'bg': 'f4', 'lpx': 'f4','lpy': 'f4',
                        'ellipticity': 'f4', 'net_gradient': 'f4', 'd_zcalib': 'f4', 'group': 'u4'})
        df2 = df.reindex(columns = ['frame', 'x', 'y', 'z', 'photons', 'sx', 
                                    'sy', 'bg', 'lpx', 'lpy', 'ellipticity', 'net_gradient', 'd_zcalib', 'group'], fill_value=1)
            
        path = os.path.split(filename)[0] + "/"
        filename_old = os.path.split(filename)[1]
        filename_new = '%s_ClusterD%s_%d_%s.hdf5' %(filename_old[:-5], threshold_radius_str, cluster_size_threshold, str(radius_z))
        tools.picasso_hdf5(df2, filename_new, filename_old, path)
    
    
    '''
    Generating hdf5 file for picasso render with the weighted centers as loclaizaitons
    '''
    grouped = df2.groupby("group")

    # Test if every cluster is a Gaussian (Shapiro Wilk test)
    from scipy.stats import shapiro
    shapiro_df = (grouped.apply(lambda x: pd.Series(shapiro(x), index=['W', 'P'])).reset_index())
    
    
    x_av_wtd = grouped.apply(lambda x: np.average(x['x'],weights=1/x['lpx']/x['lpx']))
    x_av_wtd.name = "x_av_wtd"
    y_av_wtd  = grouped.apply(lambda x: np.average(x['y'],weights=1/x['lpy']/x['lpy']))
    y_av_wtd.name = "y_av_wtd"
    if flag_3D:
        z_av_wtd = grouped.apply(lambda x: np.average(x['z'])/pl)
        z_av_wtd.name = "z_av_wtd"
    group_size = grouped.size()
    group_size.name = "group_size"


    def error_sums_wtd(group, x, lpx):
        x = group[x]
        w = 1/group[lpx]**2
        #gr_size = group.size()
        #x_av = (w * x).sum() / w.sum()
        #return (w * (x - (w * x).sum() / w.sum())**2).sum() / w.sum() * gr_size/(gr_size-1)
        return (w * (x - (w * x).sum() / w.sum())**2).sum() / w.sum()
    
    
    
    sed_x = np.sqrt(grouped.apply(error_sums_wtd, "x", "lpx")/(group_size - 1))
    sed_y = np.sqrt(grouped.apply(error_sums_wtd, "y", "lpy")/(group_size - 1))
    sed_xy = np.mean([sed_x, sed_y], axis = 0)

    group_means = grouped.mean()

    data3_frames = group_means['frame'].values.tolist()
    data3_x = x_av_wtd.values.tolist()
    data3_y = y_av_wtd.values.tolist()
    if flag_3D:
        data3_z = z_av_wtd.values.tolist()
    data3_photons = group_means['photons'].values.tolist()
    data3_sx = group_means['sx'].values.tolist()
    data3_sy = group_means['sy'].values.tolist()
    data3_bg = group_means['bg'].values.tolist()
    data3_lpx = sed_xy.tolist()
    data3_lpy = sed_xy.tolist()
    if flag_3D:
        data3_lpz = 2*sed_xy
    data3_group = np.full(shape = len(data3_lpy), fill_value = np.mean(np.array(data['group']))) # group of origami, not of clusters in origami
    data3_n = group_size.values.tolist()
    
    '''
    Generating hdf5 file for picasso render
    '''

    try:
        os.mkdir(os.path.split(filename)[0] + '/AdditionalOutputs')
    except OSError:
        print ("transf_overview folder already exists")

    if not flag_3D:
        data = {'frame': data3_frames, 'x': data3_x, 'y': data3_y, 
                'photons': data3_photons, 'sx': data3_sx, 'sy': data3_sy, 
                'bg': data3_bg, 'lpx': data3_lpx,'lpy': data3_lpy, 
                'group': data3_group, 'n': data3_n}
        
        df = pd.DataFrame(data, index=range(len(data3_x)))
        df = df.astype({'frame': 'u4', 'x': 'f4', 'y': 'f4', 
                        'photons': 'f4', 'sx': 'f4', 'sy': 'f4',
                        'bg': 'f4', 'lpx': 'f4','lpy': 'f4',
                        'group': 'u4', 'n': 'u4'})
        df3 = df.reindex(columns = ['frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy', 'group', 'n'], fill_value=1)

        path = os.path.split(filename)[0] + "/"
        filename_old = os.path.split(filename)[1]
        filename_new = '%s_resi_%s_%d.hdf5' % (filename_old[:-5], threshold_radius_str, cluster_size_threshold)
        tools.picasso_hdf5(df3, filename_new, filename_old, path)

    else:
        data = {'frame': data3_frames, 'x': data3_x, 'y': data3_y, 'z': data3_z,
                'photons': data3_photons, 'sx': data3_sx, 'sy': data3_sy, 
                'bg': data3_bg, 'lpx': data3_lpx,'lpy': data3_lpy, 'lpz': data3_lpz,
                'group': data3_group, 'n': data3_n}
        
        df = pd.DataFrame(data, index=range(len(data3_x)))
        df = df.astype({'frame': 'u4', 'x': 'f4', 'y': 'f4', 'z': 'f4', 
                        'photons': 'f4', 'sx': 'f4', 'sy': 'f4',
                        'bg': 'f4', 'lpx': 'f4','lpy': 'f4', 'lpz': 'f4',
                        'group': 'u4', 'n': 'u4'})
        df3 = df.reindex(columns = ['frame', 'x', 'y', 'z', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy', 'lpz', 'group', 'n'], fill_value=1)

        path = os.path.split(filename)[0] + "/"
        filename_old = os.path.split(filename)[1]
        filename_new = '%s_resi_%s_%d_%s.hdf5' %(filename_old[:-5], threshold_radius_str, cluster_size_threshold, str(radius_z))
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
    if not flag_3D:
        np.savez('%s_varsD%s_%d' %(filename[:-5], threshold_radius_str, cluster_size_threshold),
             data2_x=data2_x,data2_y=data2_y, data2_frames=data2_frames, data2_group=data2_group,
             new_com_x_cluster=data3_x, new_com_y_cluster=data3_y,
             amountOfNeighbors_data=amountOfNeighbors_data, x_coords=x_coords, y_coords=y_coords) 
    else:
        np.savez('%s_varsD%s_%d_%s' %(filename[:-5], threshold_radius_str, cluster_size_threshold, str(radius_z)),
             data2_x=data2_x,data2_y=data2_y, data2_z=data2_z, data2_frames=data2_frames, data2_group=data2_group,
             new_com_x_cluster=data3_x, new_com_y_cluster=data3_y,
             new_com_z_cluster=data3_z, amountOfNeighbors_data=amountOfNeighbors_data, 
             x_coords=x_coords, y_coords=y_coords, z_coords=z_coords)
