import numpy as np

import DistancesP2Vec


class NucleiConnectSingle:
    """Only class, does all the job."""
    def __init__(self, ctrs, labbs, idx_ref, t1, dist_thr):
        self.ctrs      =  np.round(ctrs).astype(np.int32)
        self.labbs     =  labbs
        self.idx_ref   =  idx_ref
        self.dist_thr  =  dist_thr
        self.t1        =  t1

        labbs2  =  np.zeros(labbs.shape)
        t_tot   =  self.labbs[:, 0, 0].size

        if self.ctrs[self.t1, 0, self.idx_ref] != 0:
            labbs2[self.t1, :, :]  +=  (labbs[self.t1, :, :] == labbs[self.t1, self.ctrs[self.t1, 0, self.idx_ref], self.ctrs[self.t1, 1, self.idx_ref]])

            for t in range(self.t1, t_tot - 1):
                dist                           =  DistancesP2Vec.DistancesP2Vec(self.ctrs[t, :, self.idx_ref], self.ctrs[t + 1, :, :])
                self.ctrs[t, :, self.idx_ref]  =  np.array([0, 0])

                if dist.dists_sqrt.min() < self.dist_thr and labbs[t + 1, self.ctrs[t + 1, 0, dist.dists_sqrt.argmin()], self.ctrs[t + 1, 1, dist.dists_sqrt.argmin()]] > 0:
                    self.idx_ref  =  dist.dists_sqrt.argmin()

                    labbs2[t + 1, :, :]  +=  (labbs[t + 1, :, :] == labbs[t + 1, self.ctrs[t + 1, 0, self.idx_ref], self.ctrs[t + 1, 1, self.idx_ref]])

                else:
                    break

        self.labbs2  =  labbs2.astype(np.int32)
