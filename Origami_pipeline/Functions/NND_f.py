# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 12:28:46 2022

@author: reinhardt Rafal
"""
import numpy as _np
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN

    def nearest_neighbor(self):
        """ Gets channel for nearest neighbor analysis. """

        # choose both channels
        channel1 = self.get_channel("Nearest Neighbor Analysis")
        channel2 = self.get_channel("Nearest Neighbor Analysis")
        self._nearest_neighbor(channel1, channel2)

    def _nearest_neighbor(self, channel1, channel2):
        """
        Calculates and saves distances of the nearest neighbors between
        localizations in channels 1 and 2
        Saves calculated distances in .csv format.
        Parameters
        ----------
        channel1 : int
            Channel to calculate nearest neighbors distances
        channel2 : int
            Second channel to calculate nearest neighbor distances
        """

        # ask how many nearest neighbors
        nn_count, ok = QtWidgets.QInputDialog.getInt(
            self, "", "Number of nearest neighbors: ", 0, 1, 100
        )
        if ok:
            # extract x, y and z from both channels 
            x1 = self.locs[channel1].x
            x2 = self.locs[channel2].x
            y1 = self.locs[channel1].y
            y2 = self.locs[channel2].y
            if (
                hasattr(self.locs[channel1], "z")
                and hasattr(self.locs[channel2], "z")
            ):
                z1 = self.locs[channel1].z
                z2 = self.locs[channel2].z
            else: 
                z1 = None
                z2 = None

            # used for avoiding zero distances (to self)
            same_channel = channel1 == channel2

            # get saved file name
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self, 
                "Save nearest neighbor distances", 
                self.locs_paths[channel1].replace(".hdf5", "_nn.csv"),
                filter="*.csv",
            )
            nn = postprocess.nn_analysis(
                x1, x2, 
                y1, y2, 
                z1, z2,
                nn_count, 
                same_channel, 
            )
            # save as .csv
            np.savetxt(path, nn, delimiter=',')

def nn_analysis(
    x1, x2, 
    y1, y2, 
    z1, z2,
    nn_count, 
    same_channel, 
):
    if z1 is not None: # 3D
        input1 = _np.stack((x1, y1, z1)).T
        input2 = _np.stack((x2, y2, z2)).T
    else: # 2D
        input1 = _np.stack((x1, y1)).T
        input2 = _np.stack((x2, y2)).T
    if same_channel:
        model = NN(n_neighbors=nn_count+1)
    else:
        model = NN(n_neighbors=nn_count)
    model.fit(input1)
    nn, _ = model.kneighbors(input2)
    if same_channel:
        nn = nn[:, 1:] # ignore the zero distance
    return nn