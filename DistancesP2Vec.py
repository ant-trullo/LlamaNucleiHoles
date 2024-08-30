"""This function calculates the distance between a point (x0 = (x0_1, x0_2))
and a vector (vec, such that vec,shape = (2, n))"""

import numpy as np


class DistancesP2Vec:

    def __init__(self, x0, vec):
        self.x0   =  x0
        self.vec  =  vec

        dists       =  np.zeros(vec[0, :].size)
        dists_sqrt  =  np.zeros(vec[0, :].size)
        for i in range(dists.size):
            dists[i]       =  (x0[0] - vec[0, i]) ** 2 + (x0[1] - vec[1, i]) ** 2
            dists_sqrt[i]  =  np.sqrt(dists[i])

        self.dists       =  dists
        self.dists_sqrt  =  dists_sqrt