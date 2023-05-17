import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import cm
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, ListedColormap

path = "Please state your path"
px_size = 130
everything = pd.read_hdf(path, key = 'locs')
#print(g)

everything.y = -1*everything.y #Invert Y coordinates to adjust from Picasso-style to conventional axis direction



##############################
###    Plot 3D overview    ###
##############################

plt.style.use('dark_background')
fig, ax = plt.subplots()
ax = fig.add_subplot(projection='3d')
ax.scatter(everything.x, everything.y, everything.z)
plt.show()



################################################
###    Plot z position encoded with color    ###
################################################

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
fig.tight_layout()
colorcode = ax.scatter(everything.x, everything.y, c=everything.z, cmap='rainbow', s = 150)
ax.set_aspect('equal')
cbar = plt.colorbar(colorcode)
plt.show()
#plt.savefig('a1_colors.png', transparent=True, dpi=1000)



df_centers = everything
#print(df_centers.keys())
n = 10000
px = 130 #in nm

def resample(row, n, px):
    
    """
    Resampling of a cluster center is performed by drawing n samples of a 
    3D Gaussian around the cluster center. The covariance matrix of the 3d 
    Gaussian is defined by the standard error of the mean previously calculated
    for each cluster center and stored in lpx, lpy and lpz columns.

    Parameters
    ----------
    row : row of pandas dataframe (series)
        A row corresponds to one RESI cluster center. 
        The row contains the following columns:  
        'frame', 'x', 'y', 'z', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy',
        'lpz', 'group_input', 'n'
        x, y: pixel
        z: nm
        lpx, lpy, lpz: pixel
    n : integer
        The number of points that will be drawn from a 3D gaussian .
    px : integer
        Pixel size in nm

    Returns
    -------
    A dataframe with n+1 rows (n resampled points plus one row for the original
    cluster center). Column keys correspond to the original dataframe row. 
    Each resampled row is a duplicate of the original row except the x, y and z
    value, which were sampled from the Gaussian distribution.
    """

    n = int(row.lpx*1000*n) #Density adjustment for clearer visualization
    
    mean = [row.x, row.y, row.z]
    cov = [[row.lpx**2,0,0],[0,row.lpy**2,0],[0,0,(row.lpz*px)**2]]
    x, y, z = np.random.multivariate_normal(mean, cov, n).T
    # append the cluster center coordinate to the resampled coordinates:
    x = np.append(np.array(row.x),x) 
    y = np.append(np.array(row.y),y)
    z = np.append(np.array(row.z),z)
    
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    
    df.insert(0, 'frame', row.frame) # insert frame column before x, y and z columns
    df['photons'] = row.photons
    df['sx'] = row.sx
    df['sy'] = row.sy
    df['bg'] = row.bg
    df['lpx'] = row.lpx
    df['lpy'] = row.lpy
    df['lpz'] = row.lpz
    df['group_input'] = row.group_input
    df['n'] = row.n

    return df

result = []
for row in df_centers.itertuples():
    print(type(row))
    result.append(resample(row, n, px))
   
result = pd.concat(result, axis = 0, ignore_index = True)

#print(result.shape)
#print(result)

result_nm = result
result_nm.x = result_nm.x*130
result_nm.y = result_nm.y*130

edge_l_x = min(result_nm.x)
edge_l_y = min(result_nm.y)
edge_r_x = max(result_nm.x)
edge_u_y = max(result_nm.y)

result_nm.x = result_nm.x-edge_l_x
result_nm.y = result_nm.y-edge_l_y



#######################################
###    Plot density in greyscale    ###
#######################################

bins=np.arange(min(result_nm.x), max(result_nm.x) + 0.4, 0.4)
fig = plt.figure(figsize=(10, 10))
fig.tight_layout()
ax = fig.add_subplot(111)
hurra = ax.hist2d(result_nm.x, result_nm.y, bins=bins, cmap = "gist_gray")
ax.set_aspect('equal')
plt.show()
#plt.savefig('a1_4ang_binsize.png', transparent=True, dpi=1000)



