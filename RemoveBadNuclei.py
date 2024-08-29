"""This function removes the nuclei on the border.

Given a time series of already tracked nuclei, this function check the nuclei that
appear even ones on the border of the images and removes them.

"""


import numpy as np
from skimage.morphology import remove_small_objects
# from PyQt5 import QtWidgets

import UsefulWidgets


class RemoveBorderNuclei:
    def __init__(self, nuclei_tracked, px_brd):

        mask                                     =  np.ones(nuclei_tracked.shape, dtype='uint16')        # B&W matrix, having 1 on the border (px_brd thick)
        mask[:, px_brd:-px_brd, px_brd:-px_brd]  =  0
        idxs_rmv                                 =  np.unique(mask * nuclei_tracked)[1:]                 # multiplication of the matrix with image to have the indexes of the nuclei touching the border

        pbar  =  UsefulWidgets.ProgressBar(total1=idxs_rmv.size)
        pbar.show()
        pbar_idx  =  1

        for k in idxs_rmv:
            pbar.update_progressbar(pbar_idx)
            pbar_idx       +=  1
            nuclei_tracked  *=  (1 - (nuclei_tracked == k)).astype('bool')                                                  # removal of all these nuclei

        pbar.close()    

        self.nuclei_tracked  =  nuclei_tracked


class RemoveSmallNuclei:
    def __init__(self, nuclei_tracked, area_thr):

        steps        =  nuclei_tracked.shape[0]
        nuclei_thrd  =  np.zeros_like(nuclei_tracked)

        for t in range(steps):
            nuclei_thrd[t]  =  remove_small_objects(nuclei_tracked[t], area_thr)

        self.nuclei_thrd  =  nuclei_thrd
