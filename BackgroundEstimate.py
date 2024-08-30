"""This function calculates the intensity of the background around spots.

Input are the 3d coordinates of the spots, tracked spots for the tags and raw data.
Output are the value of the background for each spot in each time frame.
"""

import numpy as np
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels

import UsefulWidgets


def reconstruct_spots_sing_t(spots_3d_coords, zlen, xlen, ylen, t):
    """This function reconstruct the 3d detected spots from coordinates."""
    spots_sub_coords  =  spots_3d_coords[spots_3d_coords[:, 0] == t]                                                    # extract the sub matrix with the coordinates of the time t
    spots_sing_t      =  np.zeros((zlen, xlen, ylen), dtype=np.uint16)                                                  # initialize the 3D single time frame matrix
    for uu in spots_sub_coords:
        spots_sing_t[uu[1], uu[2], uu[3]]  =  1                                                                         # put 1 where the coordinates indicate that there the pixel is part of a spot (no tags for now)
    return spots_sing_t


class BackgroundEstimate:
    """This function reconstruct in 3D spots using coordinates,
    labels them using the tracked spots and construct a shell around
    each spot expanding label to measure the aberage intensity in the shell."""
    def __init__(self, spots_3d_coords, spots_tracked, green4d):

        tlen, zlen, xlen, ylen  =  green4d.shape                                                                        # read matrix shape

        pbar  =  UsefulWidgets.ProgressBar(total1=tlen)
        pbar.update_progressbar(0)
        pbar.show()

        t_lbl_avints            =  []                                                                                   # initialize a list to store final info
        for tt in range(tlen):                                                                                          # for each time frame
            pbar.update_progressbar(tt)
            spots_3d_singtime   =  reconstruct_spots_sing_t(spots_3d_coords, zlen, xlen, ylen, tt)                      # reconstruct the 3D time point with coordinates
            spots_3d_singtime  *=  spots_tracked[tt].astype(np.uint16)                                                  # multiply to properly label in 3D
            cages               =  expand_labels(spots_3d_singtime, 5) - expand_labels(spots_3d_singtime, 3)            # build the cages: use expand label twice with different iterations and subtract to have the shells surrounding the spots
            rgp_cages           =  regionprops_table(cages, green4d[tt], properties=["label", "intensity_image", "area"])   # regionprops to measure volume and intensity and store the label
            for cnt, ll in enumerate(rgp_cages["label"]):
                t_lbl_avints.append([tt, ll, np.sum(rgp_cages["intensity_image"][cnt] / rgp_cages["area"][cnt])])       # time, label, tot intensity and volume are stored in the list

        pbar.close()

        self.cages_tli  =  np.asarray(t_lbl_avints)                                                                     # list is transformed into an array since it will be easier to isolate values


class BackgroundEstimateFillingGaps:
    """This function reconstruct in 3D spots using coordinates,
    labels them using the tracked spots and construct a shell around
    each spot expanding label to measure the aberage intensity in the shell.
    In case of mising spots here, we use the cage in the previous frame."""
    def __init__(self, spots_3d_coords, spots_tracked, green4d):

        tlen, zlen, xlen, ylen  =  green4d.shape                                                                        # read matrix shape

        spts_tags  =  np.unique(spots_tracked[spots_tracked != 0])                                                      # tags list of all the spots
        for spt_tag in spts_tags:                                                                                       # for each tag
            # print(spt_tag)
            sing_spt  =  spots_tracked == spt_tag                                                                       # isolate the spot
            if (np.sum(np.sign(np.sum(sing_spt, axis=(1, 2)))) / tlen) < 0.2:                                           # check if it is present in less than the 20% of the frames (20 is very arbitrary)
                spots_tracked  *=  (1 - sing_spt).astype(np.uint32)                                                     # if it is the case, remove the spot

        spts_tags  =  np.unique(spots_tracked[spots_tracked != 0])                                                      # re-extract the list of tag for the survived spots

        pbar  =  UsefulWidgets.ProgressBar(total1=tlen)
        pbar.update_progressbar(0)
        pbar.show()

        t_lbl_avints            =  []                                                                                                        # initialize a list to store final info
        for tt in range(tlen):                                                                                                               # for each time frame
            pbar.update_progressbar(tt)
            spots_3d_singtime   =  reconstruct_spots_sing_t(spots_3d_coords, zlen, xlen, ylen, tt)                                           # reconstruct the 3D time point with coordinates
            spots_3d_singtime  *=  spots_tracked[tt].astype(np.uint16)                                                                       # multiply to properly label in 3D
            cages               =  expand_labels(spots_3d_singtime, 5) - expand_labels(spots_3d_singtime, 3)            # build the cages: use expand label twice with different iterations and subtract to have the shells surrounding the spots
            rgp_cages           =  regionprops_table(cages, green4d[tt], properties=["label", "intensity_image", "area"])                    # regionprops to measure volume and intensity and store the label
            for cnt, ll in enumerate(rgp_cages["label"]):
                t_lbl_avints.append([tt, ll, np.sum(rgp_cages["intensity_image"][cnt] / rgp_cages["area"][cnt])])                             # time, label, tot intensity and volume are stored in the list

        pbar.close()

        pbar1  =  UsefulWidgets.ProgressBar(total1=spts_tags.size)
        pbar1.update_progressbar(0)
        pbar1.show()

        for mm, uu in enumerate(spts_tags):                                                                                                 # for each spot
            pbar1.update_progressbar(mm)
            # print(mm)
            prof       =  np.sign(np.sum(spots_tracked == uu, axis=(1, 2)))                                                                 # on off activation profile
            i_start    =  np.where(prof == 1)[0][0]                                                                                         # search the first frame in which the spot is present
            i_no_spot  =  np.where(prof == 0)[0]                                                                                            # search frames in which the spot is absent
            i_no_spot  =  i_no_spot[i_no_spot > i_start]                                                                                    # remove the frames without spot before the first activation
            for ii in i_no_spot:                                                                                                            # for each of them
                cc  =  ii - 1                                                                                                               # set 'cc' as the firts frame number preceeding where the spot is present
                while prof[cc] == 0:
                    cc  -=  1
                spots_3d_singtime   =  reconstruct_spots_sing_t(spots_3d_coords, zlen, xlen, ylen, cc)                                    # reconstruct the 3D time point with coordinates in frame cc
                spots_3d_singtime  *=  (spots_tracked == uu)[cc].astype(np.uint16)                                                        # multiply to isolate the specific spot we are searching for
                cage                =  expand_labels(spots_3d_singtime, 5) - expand_labels(spots_3d_singtime, 3)     # build the cages: use expand label twice with different iterations and subtract to have the shells surrounding the spots
                rgp_cage            =  regionprops_table(cage, green4d[ii], properties=["label", "intensity_image", "area"])              # regionprops to measure volume and intensity and store the label: the cages is calculated from a spot in time cc but the intensity is calculated on the raw data at frame ii
                t_lbl_avints.append([ii, uu, np.sum(rgp_cage["intensity_image"][0] / rgp_cage["area"][0])])                               # time, label, tot intensity and volume are stored in the list

        pbar1.close()

        self.cages_tli  =  np.asarray(t_lbl_avints)                                                                     # list is transformed into an array since it will be easier to isolate values
