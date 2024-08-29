"""This function detects spots of the 4D (time-x-y-z) stack.

This manages the detection in a multiprocessing implementation.
"""

from importlib import reload
import multiprocessing
import numpy as np

import SpotsDetection3D


class SpotsDetection3DMultiCore:
    """Only class, does all the job"""
    def __init__(self, green4d, spots3d_thr, vol_thr, gauss_kernelsize_value):

        reload(SpotsDetection3D)
        cpu_ow   =  multiprocessing.cpu_count()
        t_chops  =  1 + green4d.shape[0] // cpu_ow

        a  =  []
        for t in range(cpu_ow - 1):
            a.append(green4d[t * t_chops:(t + 1) * t_chops, :, :, :])           # in the multiprocessing pool each core will work on a certain number of frames: here we chop the frames

        a.append(green4d[(t + 1) * t_chops:, :, :, :])
        job_args  =  []
        for k in range(cpu_ow):
            job_args.append([a[k], spots3d_thr, vol_thr, gauss_kernelsize_value])

        pool     =  multiprocessing.Pool()
        results  =  pool.map(SpotsDetection3D.SpotsDetection3D, job_args)
        pool.close()

        spots_ints    =  results[0].spots_ints
        spots_vol     =  results[0].spots_vol
        # spots_lbls    =  results[0].spots_lbls
        spots_coords  =  results[0].spots_coords
        spots_tzxy    =  results[0].spots_tzxy
        for k in range(1, len(results)):
            if results[k].spots_vol.shape[0] != 0:
                spots_ints    =  np.concatenate((spots_ints, results[k].spots_ints), axis=0)
                spots_vol     =  np.concatenate((spots_vol, results[k].spots_vol), axis=0)
                # spots_lbls    =  np.concatenate((spots_lbls, results[k].spots_lbls), axis=0)
                spots_coords  =  np.concatenate([spots_coords, results[k].spots_coords + np.array([t_chops * k, 0, 0, 0])], axis=0)
                spots_tzxy    =  np.concatenate((spots_tzxy, results[k].spots_tzxy + np.array([t_chops * k, 0, 0, 0])), axis=0)

        self.spots_ints    =  spots_ints
        self.spots_vol     =  spots_vol
        # self.spots_lbls    =  spots_lbls
        self.spots_coords  =  spots_coords
        self.spots_tzxy    =  spots_tzxy
