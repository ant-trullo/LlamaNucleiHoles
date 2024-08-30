"""This function loads and concatenates .czi filedata as they come from microscope.

Taking .czi filenames as input, the output are the concatenated matrices of the
maximum intensity projection of red and green channels plus the green channel in
4D (because of 3D detection purpouses). Matrices are also flipped and rotate to
have a visualization conform to ImageJ standards. In order to avoid to use 3D
and 4D matrices for concatenation (cumbersome to handle) the matrices are
always reshaped in a 1D vector, concatenated and reshaped at the end into
a matrix shape.
"""


# import multiprocessing
import numpy as np
from czifile import CziFile

import LoadCzi5D
import UsefulWidgets


class MultiLoadCzi5D:
    """Core of multi loading function"""
    def __init__(self, fnames, nucs_spts_ch):

        # fnames           =  fnames_chs[0]
        # nucs_spts_ch     =  fnames_chs[1]
        # fnames  =  natsorted(fnames)
        time_step_value  =  None
        pix_size_x       =  None
        pix_size_z       =  None

        if len(fnames) > 0:                                                             # it can be zero when used in multiprocessing
            t_steps_done  =  False                                                      # flag for time steps reading
            mt_buff       =  LoadCzi5D.LoadCzi5D(str(fnames[0]), nucs_spts_ch)          # read first .czi file

            imarray_red    =  mt_buff.red_mtx                                           # separate its different channels
            imarray_green  =  mt_buff.green_mtx
            green4d        =  mt_buff.green4d
            red4d          =  mt_buff.red4d

            if len(mt_buff.red_mtx.shape) == 2:                                         # store info about size: if a file has just one time frame, it will have one dimension less
                time_steps           =  1
                z_steps, xlen, ylen  =  green4d.shape
            else:
                time_steps, z_steps, xlen, ylen  =  green4d.shape
                with CziFile(str(fnames[0])) as czi:
                    for attachment in czi.attachments():
                        if attachment.attachment_entry.name == 'TimeStamps':
                            timestamps  =  attachment.data()
                            break
                    else:
                        raise ValueError('TimeStamps not found')

                time_step_value  =  np.round(timestamps[1] - timestamps[0], 2)    # time step value
                t_steps_done     =  True                                          # prevent to read it again from other files (useless)

            imarray_red    =  imarray_red.reshape(imarray_red.size)                     # reshape all the matrix we declare into a 1D vector
            imarray_green  =  imarray_green.reshape(imarray_green.size)
            green4d        =  green4d.reshape(green4d.size)
            red4d          =  red4d.reshape(red4d.size)

            if len(fnames) > 1:
                for s in range(1, len(fnames)):                                         # reading further files (if any)

                    mt_buff  =  LoadCzi5D.LoadCzi5D(str(fnames[s]), nucs_spts_ch)
                    if len(mt_buff.red_mtx.shape) == 2:                                 # store info about the matrix size
                        t_steps_bff  =  1
                    else:
                        t_steps_bff  =  mt_buff.green4d.shape[0]

                        if t_steps_done is False:                                       # read the time step value on the first file that has more than 1 time frame
                            with CziFile(str(fnames[s])) as czi:
                                for attachment in czi.attachments():
                                    if attachment.attachment_entry.name == 'TimeStamps':
                                        timestamps = attachment.data()
                                        break
                                else:
                                    raise ValueError('TimeStamps not found')

                            time_step_value  =  np.round(timestamps[1] - timestamps[0], 2)    # time step value
                            t_steps_done     =  True                                          # prevent to read it again from other files (useless)

                    imarray_red    =  np.append(imarray_red, mt_buff.red_mtx.reshape(mt_buff.red_mtx.size))                      # append new data to previous
                    imarray_green  =  np.append(imarray_green, mt_buff.green_mtx.reshape(mt_buff.green_mtx.size))
                    green4d        =  np.append(green4d, mt_buff.green4d.reshape(mt_buff.green4d.size))
                    red4d        =  np.append(red4d, mt_buff.red4d.reshape(mt_buff.red4d.size))

                    time_steps  +=  t_steps_bff

            imarray_red    =  imarray_red.reshape((time_steps, xlen, ylen))                                                     # final reshape of the matrioces
            imarray_green  =  imarray_green.reshape((time_steps, xlen, ylen))
            green4d        =  green4d.reshape((time_steps, z_steps, xlen, ylen))
            red4d          =  red4d.reshape((time_steps, z_steps, xlen, ylen))

            imarray_red    =  np.rot90(imarray_red, axes=(1, 2))[:, ::-1, :]                                                    # rotation to adapt to the imageJ format
            imarray_green  =  np.rot90(imarray_green, axes=(1, 2))[:, ::-1, :]
            green4d        =  np.rot90(green4d, axes=(2, 3))[:, :, ::-1, :]
            red4d          =  np.rot90(red4d, axes=(2, 3))[:, :, ::-1, :]

            a      =  CziFile(str(fnames[0]))                                                                                   # read info about pixel size
            b      =  a.metadata()
            start  =  b.find("ScalingZ")
            end    =  b[start + 9:].find("ScalingZ")

            if start != -1:
                pix_size_z  =  np.round(float(b[start + 9:start + 7 + end]) * 1000000, 4)
                start       =  b.find("ScalingX")
                end         =  b[start + 9:].find("ScalingX")
                pix_size_x  =  np.round(float(b[start + 9:start + 7 + end]) * 1000000, 4)
            elif start == -1:                                                                   # in case metadata file are not easy to read
                start2  =  b.find('<Distance Id="X"')
                if start2 != -1:
                    kk          =  b[start2:].find('</Value>')
                    pix_size_x  =  np.round(float(b[start2 + 35:start2 + kk]) * 1000000, 4)
                    cc          =  b.find('<Distance Id="Z">')
                    cc1         =  b[cc:].find('<Value>')
                    cc2         =  b[cc:].find('</Value>')
                    pix_size_z  =  np.round(float(b[cc + cc1 + 7:cc + cc2]) * 1000000, 4)
                elif start2 == -1:                                                               # in case metadata file are not easy to read
                    [pix_size_x, pix_size_z]  =  UsefulWidgets.SetPixelSize().getPixelsValues()

            self.time_steps       =  time_steps
            self.pix_size_x       =  pix_size_x
            self.pix_size_z       =  pix_size_z
            self.time_step_value  =  time_step_value
            self.imarray_red      =  imarray_red
            self.imarray_green    =  imarray_green
            self.green4d          =  green4d
            self.red4d            =  red4d
