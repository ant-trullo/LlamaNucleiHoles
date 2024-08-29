"""This function writes and reads a spots matrix as a 4xN array.

For each row of the array, there is the value of the tag and the z, x, y coordinate
of all the non-zero pixels. This is very useful in case of sparse matrix.
"""

import numpy as np
from skimage.measure import regionprops_table


class SpotsMatrixSaver:
    """Function to write a spots matrix as a list of coordinates and tags"""
    def __init__(self, spts_mtx, folder, fname):

        mtx2write  =  []                                                            # initialize list
        ff         =  np.where(spts_mtx != 0)
        for hh in range(ff[0].size):
            mtx2write.append([spts_mtx[ff[0][hh], ff[1][hh], ff[2][hh]], ff[0][hh], ff[1][hh], ff[2][hh]])

        mtx2write.append([0, spts_mtx.shape[0], spts_mtx.shape[1], spts_mtx.shape[2]])   # last element is the size of the starting matrix image
        mtx2write  =  np.asarray(mtx2write)                                                     # convert to array and write
        np.save(folder + fname, mtx2write)


class SpotsMatrixReader:
    """Function to reconstruct a spots matrix starting from its list of coordinates and tags"""
    def __init__(self, folder, fname):

        mtx2build  =  np.load(folder + fname)                                                                   # load the matrix
        spts_lbls  =  np.zeros((mtx2build[-1, 1], mtx2build[-1, 2], mtx2build[-1, 3]), dtype=np.uint32)         # initialize the matrix image with the info in the matrix
        for kk in range(mtx2build.shape[0] - 1):
            spts_lbls[mtx2build[kk, 1], mtx2build[kk, 2], mtx2build[kk, 3]]  =  mtx2build[kk, 0]                # fill the matrix with the info

        self.spts_lbls  =  spts_lbls


class SpotsMatrixSaver2D:
    """Function to write a spots matrix as a list of coordinates and tags"""
    def __init__(self, spts_mtx, folder, fname):

        mtx2write  =  []                                                            # initialize list
        tlen       =  spts_mtx.shape[0]
        spts_mtx   =  spts_mtx.astype(np.uint16)
        for tt in range(tlen):
            rgp_spts  =  regionprops_table(spts_mtx[tt], properties=["label", "coords"])                                       # regionprops to create a dictionary with al the info
            for ii, oo in enumerate(rgp_spts["coords"]):
                for cc in oo:
                    mtx2write.append([rgp_spts["label"][ii], tt, cc[0], cc[1]])                                 # each element of the list is a list with label, z coord, x coord, y coord of each pixel

        mtx2write.append([0, spts_mtx.shape[0], spts_mtx.shape[1], spts_mtx.shape[2]])   # last element is the size of the starting matrix image
        mtx2write  =  np.asarray(mtx2write)                                                                             # convert to array and write
        np.save(folder + fname, mtx2write)


class SpotsMatrixReader2D:
    """Function to reconstruct a spots matrix starting from its list of coordinates and tags"""
    def __init__(self, folder, fname):

        mtx2build  =  np.load(folder + fname)                                                                   # load the matrix
        spts_lbls  =  np.zeros((mtx2build[-1, 1], mtx2build[-1, 2], mtx2build[-1, 3]), dtype=np.uint32)         # initialize the matrix image with the info in the matrix
        for kk in range(mtx2build.shape[0] - 1):
            spts_lbls[mtx2build[kk, 1], mtx2build[kk, 2], mtx2build[kk, 3]]  =  mtx2build[kk, 0]                # fill the matrix with the info

        self.spts_lbls  =  spts_lbls

