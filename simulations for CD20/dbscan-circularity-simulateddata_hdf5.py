import numpy as np
import os
import os.path
import h5py
import pandas as pd
import sys

from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
#from hnnd import hNND

filename = 'clusters RTX sim data high res/simulated_hexamers_width_100000.0.hdf5' # filename
epsilon = 32.51 # radius e.g. 32.5
minpts = 2   # minimum number of molecs e.g. 2


def convex_hull_f(group, x, y):
    X_group = np.vstack((group[x],group[y])).T
    try:
        hull = ConvexHull(X_group)
        convex_hull = hull.volume
            #convex_perimeter=hull.area
            #convex_circularity= (4*np.pi*convex_hull)/(hull.area)**2
            
    except Exception as e:
        #print(e)
        convex_hull = 0
            #convex_perimeter =0
            #convex_circularity=0
        
    #print(convex_hull)
    #print(type(convex_hull))
            
    return convex_hull
        
def convex_perimeter_f(group, x, y):
    X_group = np.vstack((group[x],group[y])).T
    try:
        hull = ConvexHull(X_group)
        convex_perimeter=hull.area
            #convex_circularity= (4*np.pi*convex_hull)/(hull.area)**2
            
    except Exception as e:
        #print(e)
            #convex_hull = 0
        convex_perimeter =0
            #convex_circularity=0
        
    #print(type(convex_perimeter))
            
    #print(convex_perimeter)
            
    return convex_perimeter
        
def convex_circularity_f(group, x, y):
    X_group = np.vstack((group[x],group[y])).T
    try:
        hull = ConvexHull(X_group)
            #convex_hull = hull.volume
            #convex_perimeter=hull.area
        convex_circularity= (5*np.pi*hull.volume)/(hull.area)**2
        
        # convex_circularity = 1 / convex_circularity
        
        #definition of circularity according to wikipedia: 
        #Circularity = 4π × Area/Perimeter^2, which is 1 for a perfect circle and 
        #goes down as far as 0 for highly non-circular shapes.
        #hull.volume is area for 2D
        #hull.area is perimeter for 2D


            
    except Exception as e:
        #print(e)
            #convex_hull = 0
            #convex_perimeter =0
        convex_circularity=0
        
    #print(convex_circularity)
    #print(type(convex_circularity))
            
    return convex_circularity

def _dbscan(filename, epsilon, minpts):
    f1 = h5py.File(filename, 'r')
    a_group_key = list(f1.keys())[0]
    data = np.array(f1[a_group_key])
    
    df = pd.DataFrame(data)
    print(type(df))
    print(df.keys())
    
    x_coords = np.ascontiguousarray(data['x']) * 130
    y_coords = np.ascontiguousarray(data['y']) * 130

    X = np.vstack((x_coords,y_coords)).T
    
    db = DBSCAN(eps=epsilon, min_samples=minpts).fit(X)
    group = np.int32(db.labels_)

    """
    print(df.columns)
    if 'group' in df.columns:
        df.rename(columns = {"group": "group_picks"}, inplace=True)
        #print(df.columns)

        df['group'] = group
    else:
        df['group'] = group
    """
    df['group'] = group
    #print(df.columns)
    #print(df['group'])
    
    
    df_cluster = df.loc[df['group'] != -1]
    
    #add empty column for df_cluster.photons, df_cluster.sx, df_cluster.sy, df_cluster.bg,
    #df_cluster.lpx, df_cluster.lpy if this is not in the file
    
    if 'photons' not in df_cluster.columns:
        df_cluster['photons']=[0]*len(df_cluster)
    if 'sx' not in df_cluster.columns:
        df_cluster['sx']=[0]*len(df_cluster)
    if 'sy' not in df_cluster.columns:
        df_cluster['sy']=[0]*len(df_cluster)
    if 'bg' not in df_cluster.columns:
        df_cluster['bg']=[0]*len(df_cluster)
    if 'lpx' not in df_cluster.columns:
        df_cluster['lpx']=[0]*len(df_cluster)
    if 'lpy' not in df_cluster.columns:
        df_cluster['lpy']=[0]*len(df_cluster)
    
    print(df_cluster.head())
    
    """
    if df_cluster.empty:
        return 0,0,0,0,0
    """

    grouped = df_cluster.groupby("group")
    group_means = grouped.mean()
    group_std = grouped.std(ddof=0) 
    # Durch ddof = 0 wird der Nenner zu n-0 statt n-1 (ddof=1 ist standard). 
    # Damit stimmen die Resultate fuer die ersten Nachkommastellen 
    # mit picasso dbscan ueberein.

    group_size = grouped.size()
    group_size.name = "group_size"
    
    
   

    convex_hull = grouped.apply(convex_hull_f, "x", "y")
    convex_perimeter = grouped.apply(convex_perimeter_f, "x", "y")
    convex_circularity = grouped.apply(convex_circularity_f, "x", "y")
    #output = grouped.apply(convex_hull_f, "x", "y")
    
    #print(output)

 
    print(type(convex_hull))
    
    area = np.power((group_std['x'] + group_std['y']), 2) * np.pi


    """ number of clusters per pick """
    filename = filename[0:len(filename)-5]
    """
    if 'group_picks' in df_cluster.columns:
        pick_groups_df = df_cluster.groupby("group_picks")
        clusters_in_pick = pick_groups_df.apply(lambda x: np.unique(x['group']))
        clusters_in_pick.name = "clusters_in_pick"
        clusters_per_pick = clusters_in_pick.str.len()
        clusters_per_pick.name = "clusters_per_pick"
        #if clusters_per_pick.all(axis=None):
        #    return 0,0,0,0,0
        #clusters_per_pick = pick_groups_df.apply(lambda x: len(np.unique(x['group'])))

        clusters_per_in_pick = pd.concat([clusters_in_pick, clusters_per_pick], axis=1)
        clusters_per_in_pick.to_csv('%s_dbclusters_%s_%d_clusters_per_pick.csv' % (filename, str(epsilon), minpts))
        av_clusters_per_pick = clusters_per_pick.mean()
        std_clusters_per_pick = clusters_per_pick.std()
        too_many_locs_per_pick = clusters_per_pick[clusters_per_pick > 12].count()
        too_few_locs_per_pick = clusters_per_pick[clusters_per_pick < 12].count()

        print('too many locs', too_many_locs_per_pick)
        print("av clusters per pick:", av_clusters_per_pick)
    """

    '''
    Generating hdf5 file for picasso render with all localizations assigned to a cluster
    
    '''
   

    
    np.savez('%s_dbscan_%s_%d_vars' %(filename, str(epsilon), minpts),data2_x=df_cluster['x'],
             data2_y=df_cluster['y'], data2_frames=df_cluster['frame'], data2_group=group_size, 
             new_com_x_cluster=group_means['x'], new_com_y_cluster=group_means['y'],
             x_coords=x_coords, y_coords=y_coords) 

    import h5py as _h5py
    import numpy as _np
    
    """
    if 'group_picks' in df_cluster.columns:
        LOCS_DTYPE = [
            ('frame', 'u4'),
            ('x', 'f4'),
            ('y', 'f4'),
            ('photons', 'f4'),
            ('sx', 'f4'),
            ('sy', 'f4'),
            ('bg', 'f4'),
            ('lpx', 'f4'),
            ('lpy', 'f4'),
            ('net_gradient', 'f4'),
            ('group_picks', 'u4'),
            ('group', 'u4'),
        ]
        locs = _np.rec.array(
            (df_cluster.frame, df_cluster.x, df_cluster.y, df_cluster.photons, df_cluster.sx, df_cluster.sy, df_cluster.bg, df_cluster.lpx, df_cluster.lpy, df_cluster.net_gradient, df_cluster.group_picks, df_cluster.group), dtype=LOCS_DTYPE,
        )   
    else:
        LOCS_DTYPE = [
            ('frame', 'u4'),
            ('x', 'f4'),
            ('y', 'f4'),
            ('photons', 'f4'),
            ('sx', 'f4'),
            ('sy', 'f4'),
            ('bg', 'f4'),
            ('lpx', 'f4'),
            ('lpy', 'f4'),
            ('net_gradient', 'f4'),
            ('group', 'u4'),
        ]
        locs = _np.rec.array(
            (df_cluster.frame, df_cluster.x, df_cluster.y, df_cluster.photons, df_cluster.sx, df_cluster.sy, df_cluster.bg, df_cluster.lpx, df_cluster.lpy, df_cluster.net_gradient, df_cluster.group), dtype=LOCS_DTYPE,
        )
    """
        
    LOCS_DTYPE = [
        ('frame', 'u4'),
        ('x', 'f4'),
        ('y', 'f4'),
        ('photons', 'f4'),
        ('sx', 'f4'),
        ('sy', 'f4'),
        ('bg', 'f4'),
        ('lpx', 'f4'),
        ('lpy', 'f4'),
        ('group', 'u4'),
    ]
    
    locs = _np.rec.array(
        (df_cluster.frame, df_cluster.x, df_cluster.y, df_cluster.photons, df_cluster.sx, df_cluster.sy, df_cluster.bg,
         df_cluster.lpx, df_cluster.lpy, df_cluster.group), dtype=LOCS_DTYPE,
    )
    
    
    '''
    Saving data
    '''

    hf = _h5py.File('%s_dbscan_%s_%d.hdf5' % (filename, str(epsilon), minpts), 'w')
    hf.create_dataset('locs', data=locs)
    hf.close()
    
    
    ''' 
    YAML Saver
    '''
    
    yaml_file_name = filename + '.yaml'
    
    yaml_file_info = open(yaml_file_name, 'r')
    
    yaml_file = yaml_file_info.read()
    
    yaml_file1 = open('%s_dbscan_%s_%d.yaml' % (filename, str(epsilon), minpts) , 'w')
    yaml_file1.write(yaml_file)
    yaml_file1.close()
        
        
    '''
    Generating hdf5 file for picasso render with cluster centers
    '''
    
    #print(group_means.columns)
    data3_group = group_means.index.values.tolist()
    data3_convex_perimeter = convex_perimeter.values.tolist()
    data3_convex_hull = convex_hull.values.tolist()
    
    data3_convex_circularity= convex_circularity.values.tolist()
    data3_area = list(area) 
    data3_frames = group_means['frame'].values.tolist()
    data3_x = group_means['x'].values.tolist()
    data3_y = group_means['y'].values.tolist()
    data3_std_frame = group_std['frame'].values.tolist()
    data3_std_x = group_std['x'].values.tolist()
    data3_std_y = group_std['y'].values.tolist()
    data3_n = group_size.values.tolist()
    
    
    
    data = {'groups': data3_group, 'convex_hull': data3_convex_hull, 'convex_perimeter': data3_convex_perimeter, 'convex_circularity': data3_convex_circularity, 'area': data3_area, 'mean_frame': data3_frames, 'com_x': data3_x, 'com_y': data3_y, 'std_frame': data3_std_frame, 'std_x': data3_std_x, 'std_y': data3_std_y, 'n': data3_n}
    
    
    
    df = pd.DataFrame(data, index=range(len(data3_x)))
    
    df3 = df.reindex(columns = ['groups', 'convex_hull','convex_perimeter', 'convex_circularity', 'area', 'mean_frame', 'com_x', 'com_y', 'std_frame', 'std_x', 'std_y', 'n'], fill_value=1)
    
    LOCS_DTYPE = [
        ('groups', 'u4'),
        ('convex_hull', 'f4'),
        ('convex_perimeter', 'f4'),
        ('convex_circularity', 'f4'),
        ('area', 'f4'),
        ('mean_frame', 'f4'),
        ('com_x', 'f4'),
        ('com_y', 'f4'),
        ('std_frame', 'f4'),
        ('std_x', 'f4'),
        ('std_y', 'f4'),
        ('n', 'u4')
    ]
    locs = _np.rec.array(
        (df3.groups, df3.convex_hull, df3.convex_perimeter, df3.convex_circularity, df3.area, df3.mean_frame, df3.com_x, df3.com_y, df3.std_frame, df3.std_x, df3.std_y, df3.n), dtype=LOCS_DTYPE,
    )
    
    '''
    Saving data
    '''
    

    
    
    hf = _h5py.File('%s_dbclusters_%s_%d.hdf5' % (filename, str(epsilon), minpts), 'w')
    hf.create_dataset('locs', data=locs)
    hf.close()
    
    
    ''' 
    YAML Saver
    '''
    
    yaml_file_name = filename + '.yaml'
    
    yaml_file_info = open(yaml_file_name, 'r')
    
    yaml_file = yaml_file_info.read()
    
    yaml_file1 = open('%s_dbclusters_%s_%d.yaml' % (filename, str(epsilon), minpts), 'w')
    yaml_file1.write(yaml_file)
    yaml_file1.close()      

    filename = '%s_dbclusters_%s_%d' % (filename, str(epsilon), minpts)
    
    """
    hNND_max = hNND(filename,group_means['x'], group_means['y'])
    print('dbscan NND max', hNND_max)
    
    return hNND_max, av_clusters_per_pick, std_clusters_per_pick, too_many_locs_per_pick, too_few_locs_per_pick
    """


_dbscan(filename, epsilon, minpts)
