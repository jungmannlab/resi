"""
    Origami_Plotter
    ----------
    
    :author: Philipp Steen, 2022
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def FindCorners(R1_table, R3_table):
	"""
	Find inner and outermost data points to normalize coordinates 
	and adjust field of view for the plot

	Parameters
	----------
	R1_table : Pandas dataframe
		All localizations from the R1 channel .hdf5
	R3_table : Pandas dataframe
		All localizations from the R3 channel .hdf5
	"""

	Minx = min(R1_table["x"].min(), R3_table["x"].min())*130 -5
	Miny = max(R1_table["y"].max(), R3_table["y"].max())*130 +5
	maxOuter = max(((max(R1_table["x"].max(), R3_table["x"].max())*130)-Minx), 
		(-(min(R1_table["y"].min(), R3_table["y"].min())*130)+Miny)) +5

	return(Minx, Miny, maxOuter)


def Transform(R1_table, R3_table, Minx, Miny, maxOuter):
	"""
	Transform coordinates from pixels to nanometers and set the bottom 
	left of the plot to 0, 0 for an understandable representation

	Parameters
	----------
	R1_table : Pandas dataframe
		All localizations from the R1 channel .hdf5
	R3_table : Pandas dataframe
		All localizations from the R3 channel .hdf5
	Minx : float
		The minimal x-coordinate as calculated via FindCorners
	Miny : float
		The minimal y-coordinate as calcualted via FindCorners
	maxOuter : float
		The maximal coordinate as calcualted via FindCorners
	"""

	R1_table["x"] = 130*R1_table["x"]-Minx
	R3_table["x"] = 130*R3_table["x"]-Minx
	R1_table["y"] = -(130)*R1_table["y"]+Miny
	R3_table["y"] = -(130)*R3_table["y"]+Miny
	return(R1_table, R3_table)


def Import(R1_c, R3_c, R1_a, R3_a):
	"""
	Imports Picasso .hdf5 files and adjusts coordinates via Transform

	Parameters
	----------
	R1_c : path
		Path to clustered (group info present) localizations from the R1 channel
	R3_c : path
		Path to clustered (group info present) localizations from the R3 channel
	R1_a : path
		Path to all localizations from the R1 channel
	R3_a : path
		Path to all localizations from the R3 channel

	"""

	R1_table = pd.read_hdf(R1_c, key = 'locs')
	R3_table = pd.read_hdf(R3_c, key = 'locs')
	R1_table.sort_values(by=['group', 'frame'])
	R3_table.sort_values(by=['group', 'frame'])

	R1_messy = pd.read_hdf(R1_a, key = 'locs')
	R3_messy = pd.read_hdf(R3_a, key = 'locs')

	Minx, Miny, maxOuter = FindCorners(R1_table, R3_table)

	R1_table, R3_table = Transform(R1_table, R3_table, Minx, Miny, maxOuter)
	R1_messy, R3_messy = Transform(R1_messy, R3_messy, Minx, Miny, maxOuter)

	return(R1_table, R3_table, R1_messy, R3_messy, maxOuter)


def DistanceFinder(R1_centers, R3_centers, max_distance):
	"""
	Finds the distances between R1 and R3 channel RESI localizations. 
	Since group numbers are inconsistent between channels, each RESI localization
	is compared against all others. The smallest value corresponds to the nearest
	neighbor from the other channel. If there is no neighbor nearer than max_distance,
	the RESI localization has no equivalent in the other channel. 

	Parameters
	----------
	R1_centers : Numpy array
		R1 RESI localization coordinates 
	R3_centers : Numpy array
		R3 RESI localization coordinates 
	max_distance : float
		The maximum distance to consider for distance measurments

	"""

	R1_indices = []
	Distances = []
	All_distances = []
	for count, value in enumerate(R1_centers):
		R1_indices.append(count)
		for j in R3_centers:
			distance = (np.sqrt((value[0]-j[0])**2 + ((value[1]-j[1])**2)))
			if math.isnan(distance) == False:
				All_distances.append(distance)
		if len(All_distances) != 0:
			best_distance = np.min(All_distances)
		else:
			best_distance = 0
		All_distances = []
		if best_distance <= max_distance:
			Distances.append((count, best_distance))
		else:
			Distances.append((count, " "))
	return(Distances)


def ResiPrec(K, NeNa):
	"""
	Returns the precision for a RESI localization based on the DNA-PAINT NeNa precision (NeNa) 
	and the number of localizations (K)

	Parameters
	----------
	K : float
		The number of localizations for a RESI spot
	NeNa : float
		The NeNa precision of the overall DNA-PAINT measurement
	"""

	if K != 0:
		return(NeNa/(math.sqrt(K)))
	else:
		return(1e20)


#RESI class

class Origami():
    def __init__(self,
    			 ori_number = 0, 
                 NeNa = 3, #nm
                 R1_c = "",
                 R3_c = "",
                 R1_a = "",
                 R3_a = "",
                 max_distance = 5,
                 output_location = "./outputs/"): #default path
        
        self.ori_number = ori_number
        self.NeNa = NeNa
        self.R1_c = R1_c
        self.R3_c = R3_c
        self.R1_a = R1_a
        self.R3_a = R3_a
        self.max_distance = max_distance
        self.output_location = output_location

        self.R1_table = []
        self.R3_table = []
        self.R1_messy = []
        self.R3_messy = []

        self.R1_table, self.R3_table, self.R1_messy, self.R3_messy, self.maxOuter = Import(R1_c, R3_c, R1_a, R3_a)

        self.R1_RESI = []
        self.R3_RESI = []
        self.R1_prec = []
        self.R3_prec = []

        self.distances = []

        self.fig = []
        self.ax = []

    def CalcRESI(self):
        """
        Calculates the centers (coordinates) of all grouped localizations from R1_table 
        and R3_table as well as their precision (via ResiPrec).
        Also calcualtes the distances (smaller than max_distance) from R1 RESI 
        localizations to their nearest neighbors in the R3 channel.
        """

        R1_RESI = []
        R3_RESI = []
        R1_prec = []
        R3_prec = []

        #These pick (group) numbers correspond to R1 and R3 sites
        for i in range(max(self.R1_table["group"].max(), self.R3_table["group"].max()) + 1):
        	R1 = (self.R1_table[self.R1_table["group"] == i])
        	Avg_R1 = R1["x"].mean(), R1["y"].mean()

        	R3 = (self.R3_table[self.R3_table["group"] == i])
        	Avg_R3 = R3["x"].mean(), R3["y"].mean()

        	if math.isnan(Avg_R1[0]) == False:
        		R1_RESI.append(Avg_R1)
        		R1_prec.append(ResiPrec(len(R1), self.NeNa))
        	if math.isnan(Avg_R3[0]) == False:
        		R3_RESI.append(Avg_R3)
        		R3_prec.append(ResiPrec(len(R3), self.NeNa))

        #Turn the lists into numpy arrays for better usability
        self.R1_RESI = np.asarray(R1_RESI)
        self.R3_RESI = np.asarray(R3_RESI)
        self.R1_prec = np.asarray(R1_prec)
        self.R3_prec = np.asarray(R3_prec)

        self.distances = np.asarray(DistanceFinder(self.R1_RESI, self.R3_RESI, self.max_distance))

    def PlotRESI(self, Locations, Sizes, Fill_Color, Edge_Color, Opacity, Show_Labels, Textcolor, Whichside, Labels):
    	"""
        Plots the RESI localizations at their respective coordinates. The sizes of
        the dots is determined by the RESI precision. The distances between nearest
        neighbors of different channels are displayed next to the R1 localizations.

        Parameters
        ----------
        Locations : Numpy Array
			Coordinates of the RESI localizations
        Sizes : Numpy Array
			Sizes for the RESI localizations (RESI precision)
        Fill_Color : String (must be color)
			Color to fill the RESI points
        Edge_Color : String (must be color)
			Color to border the RESI points
        Opacity : Float (must be between 0 and 1)
			Opacity of the RESI points
        Show_Labels : Boolean
			Whether or not to display distances
        Textcolor : String (must be color)
			Which color to display distances in 
        Whichside : String (must be "left" or "right")
			Which side of the RESI points to display the distances on
        Labels : List of strings
        	The distances (text to be displayed)
        """

    	for count, value in enumerate(Locations):
    		if math.isnan(value[0]) or math.isnan(value[1]):
    			print("NaN error")
    		else:
    			draw_circle = plt.Circle((value[0], value[1]), Sizes[count],
    				alpha=Opacity, facecolor = Fill_Color, edgecolor = Edge_Color)
    			self.ax.add_artist(draw_circle)
    		if Show_Labels==True:
    			try:
    				label_to_display = float(Labels[count])
    				label_to_display = str(round(label_to_display, 2))+" nm"
    			except:
    				label_to_display = Labels[count]

    			if Whichside=="left":
    				self.ax.text(value[0]-11, value[1]-1, label_to_display, color = Textcolor)
    			else:
    				self.ax.text(value[0]+1.5, value[1]-1, label_to_display, color = Textcolor)


    def AutoPlotter(self, color, show, save):
    	"""
        Plots the RESI localizations (via PlotRESI) as well as the "raw" localizations
        the RESI localizations are based on. 

        Parameters
        ----------
        color : String (must be "black" or "white")
			The desired background color of the plots. The text color etc. are automatically
			adjusted accordingly. 
        show : Boolean
			Whether or not to display the plots as the code runs
        save : Boolean
			Whether or not to save the plots
        """

    	size_inches = 8
    	dpi = 150
    	self.fig, self.ax = plt.subplots(figsize=(size_inches, size_inches), dpi=dpi)
    	self.ax.axis('equal')
    	self.ax.set_ylim(0, self.maxOuter)
    	self.ax.set_xlim(0, self.maxOuter)
    	if color == "black":
    		self.ax.set_facecolor("black")
    		alpha1 = 0.15
    		alpha2 = 1
    		edge = "white"
    		textcol = "white"
    	elif color == "white":
    		alpha1 = 0.2
    		alpha2 = 0.4
    		textcol = "black"
    		edge = "black"

    	self.ax.scatter(self.R1_table["x"], self.R1_table["y"], s=0.5, alpha=alpha1, color = "orange")
    	self.ax.scatter(self.R3_table["x"], self.R3_table["y"], s=0.5, alpha=alpha1, color = "teal")
    	self.ax.scatter(self.R1_messy["x"], self.R1_messy["y"], s=0.4, alpha=alpha1, color = "red")
    	self.ax.scatter(self.R3_messy["x"], self.R3_messy["y"], s=0.4, alpha=alpha1, color = "lightblue")

    	labels = self.distances[:,1]

    	self.PlotRESI(self.R1_RESI, self.R1_prec, "red", edge, alpha2, True, textcol, "right", Labels = labels)
    	self.PlotRESI(self.R3_RESI, self.R3_prec, "blue", edge, alpha2, False, textcol, "right", Labels = labels)
    	
    	title = "Origami "+ str(self.ori_number)
    	plt.title(title)
    	plt.tight_layout()
    	
    	if show==True:
    		plt.show()

    	if save==True:
    		title_2 = title+".pdf"
    		where_to = os.path.join(self.output_location, title_2)
    		plt.savefig(where_to, transparent=False, bbox_inches='tight')
    	plt.close()



#Cycle through all files of interest

def Cycler(parent_folder, R1_clusters, R3_clusters, R1_all, R3_all, number):
	"""
	Generates the filenames necessary to import the data.

	Parameters
	----------
	parent_folder : String (path)
		Path to the folder containing all data
	R1_clusters : String
		Filename-components of the clustered R1 localizations
	R3_clusters : String
		Filename-components of the clustered R3 localizations
	R1_all : String
		Filename-components of all R1 localizations
	R3_all : String
		Filename-components of all R3 localizations
	number: String
		The numbers of the origami (0 to ...) to complete the filenames
	"""
	R1_c_file = R1_clusters[0]+number+R1_clusters[1]
	R3_c_file = R3_clusters[0]+number+R3_clusters[1]
	R1_a_file = R1_all[0]+number+R1_all[1]
	R3_a_file = R3_all[0]+number+R3_all[1]
	path_R1_c = os.path.join(parent_folder, R1_c_file)
	path_R3_c = os.path.join(parent_folder, R3_c_file)
	path_R1_a = os.path.join(parent_folder, R1_a_file)
	path_R3_a = os.path.join(parent_folder, R3_a_file)

	return(path_R1_c, path_R3_c, path_R1_a, path_R3_a)


parent_folder = "/Volumes/pool-miblab4/users/steen/z.microscopy_raw/02_Project_RESI/211221_RESI_newVoyager_FusionBT_R1R3_nodel/resi-analysis_2/"

R1_clusters = ["R1_picked_ori", "_ClusterD4_50.hdf5"]
R3_clusters = ["R3_picked_ori", "_aligned_ClusterD4_50.hdf5"]

R1_all = ["R1_picked_ori", ".hdf5"]
R3_all = ["R3_picked_ori", "_aligned.hdf5"]

#numbers = ["0", "1", "3", "5", "6", "14", "12", "11", "16", "18", "23", "21", "29", "27", "35", "34", "33", "40", "39", "38", "36", "41"]

#numbers = (map(str, range(92))) #Adjust depending on how many origami you have picked.

numbers = ["2", "3"]

#out_path = "./outputs/"

out_path = "/Users/steen/Projects/RESI/figures_out/"

for index, value in enumerate(numbers):
	R1_c, R3_c, R1_a, R3_a = Cycler(parent_folder, R1_clusters, R3_clusters, R1_all, R3_all, value)
	f = Origami(value, 2.5, R1_c, R3_c, R1_a, R3_a, 5, out_path)
	f.CalcRESI()
	f.AutoPlotter("black", show = False, save = True)

