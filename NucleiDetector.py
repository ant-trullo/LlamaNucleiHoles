"""This function detects nuclei thanks to a minimum intensity projection.

Nuclei here are obtained from the holes of singals in green channel.
Input is the TxZxXxY, out put is the matrix-video of the detected nuclei.
"""

import numpy as np
from skimage.filters import gaussian,hessian, threshold_otsu
from skimage.morphology import remove_small_holes, label, binary_erosion, disk, closing
from skimage.measure import regionprops_table
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, expand_labels
from scipy import ndimage

import UsefulWidgets


class NucleiDetector:
    """Only class, does all the job."""
    def __init__(self, green4d):

        tlen, zlen, xlen, ylen  =  green4d.shape                                                                        # shape of the input matrix
        green_minp              =  np.zeros((tlen, xlen, ylen), dtype=green4d.dtype)                                    # initialize the image-matrix for the intensity projection (sum in z)

        pbar  =  UsefulWidgets.ProgressBarTriple(total1=tlen, total2=tlen, total3=tlen)
        pbar.show()
        pbar.update_progressbar1(0)

        mxxs  =  []
        for tt in range(tlen):                                                                                          # for each time step
            pbar.update_progressbar1(tt + 1)
            # z_prof          =  np.sum(green4d[tt], axis=(1, 2))
            # mxx             =  argrelextrema(z_prof, np.greater)[0]
            # if mxx.size == 1:
            #     mxx  =  np.array([0, mxx[0]])
            # print(mxx)

            # green_bff       =  gaussian(green4d[tt, mxx[0]:mxx[1]].astype(np.float32), 1.5)                                            # gaussian smoothing
            mxx             =  [16, 20]
            mxxs.append(mxx)
            green_bff       =  gaussian(green4d[tt, mxx[0]:mxx[1]].astype(np.float32), 1.5)                                            # gaussian smoothing
            green_minp[tt]  =  green_bff.min(0)                                                                         # sum over z
            # green_minp[tt]  =  green_bff.sum(0)                                                                         # sum over z

        bw_nucs  =  np.zeros((tlen, xlen, ylen), dtype=bool)                                                            # initialize the black&white video-matrix
        for uu in range(tlen):                                                                                          # for each time frame
            pbar.update_progressbar2(uu + 1)
            bff          =  hessian(green_minp[uu])                                                                     # hessian filter to enhance the borders
            bw_nucs[uu]  =  (bff > .5)                                                                                  # for the way results is, the threshold is 0.5
            bw_nucs[uu]  =  remove_small_holes(bw_nucs[uu], 100)                                                        # remove holes (TS makes holes in the nucleus calc)
            bw_nucs[uu]  =  gaussian(bw_nucs[uu].astype(float), .5)                                                     # gaussian filter to smooth (kernel .5 works)
            val          =  threshold_otsu(bw_nucs[uu])                                                                 # otsu threshold on the gaussian filtered image
            bw_nucs[uu]  =  (bw_nucs[uu] > val) * 1                                                                     # thresholding
            bw_nucs[uu]  =  1 - bw_nucs[uu]                                                                             # image bw flip since borders are white
            bw_nucs[uu]  =  remove_small_holes(bw_nucs[uu], 400)                                                        # remove the hole left by the TSs
            bw_nucs[uu]  =  closing(bw_nucs[uu], disk(5))                                                               # closure to make the smooth calcs borders
            bw_nucs[uu]  =  binary_erosion(bw_nucs[uu], disk(6))                                                        # erosion (here the purpouse is to sample inside nuclei, so if calcs are shrinked it is fine)

        nucs_lbld  =  np.zeros(bw_nucs.shape, dtype=np.uint32)                                                          # initialize the matrix of labeled nuclei
        for oo in range(tlen):                                                                                          # for each time frame
            nucs_lbld[oo]  =  label(bw_nucs[oo], connectivity=1)                                                        # label frame by frame (is a 2D labeling)

        nucs2work  =  np.zeros_like(nucs_lbld)                                                                          # initialize the matrix of the nuclei to work on (watershed)
        # areas_list  =  []
        for yy in range(tlen):                                                                                          # for each time frame
            rgp_bff  =  regionprops_table(nucs_lbld[yy], properties=("label", "area"))                                  # regionpros of the nuclei
            gg       =  np.where(rgp_bff["area"] > 800)[0]                                                              # threshold size for a nucleus to be correctly segmented (THIS PART CAN BE IMPROOVED WITH A GAUSSIAN FITTING ON THE HISTOGRAM OF THE AREAS DISTRIBUTION)
            for gg_s in gg:                                                                                             # all nuclei with a surface higher than the threshold are nuclei to segment (2 or more touching
                nucs2work[yy]  +=  (nucs_lbld[yy] == rgp_bff["label"][gg_s]) * np.uint32(rgp_bff["label"][gg_s])        # add the nuclei to segment into the matrix nucs2work
            # areas_list  +=  list(rgp_bff["area"])

        nucs_lbld  *=  (1 - np.sign(nucs2work))                                                                         # remove the nuclei which segmentation must be correct from the segmented nuclei matrix

        for hh in range(tlen):                                                                                          # for each time frame
            pbar.update_progressbar3(hh + 1)
            nucs_bff      =  binary_erosion(np.sign(nucs2work[hh])) * nucs2work[hh]                                     # binary erosion of the nuclei to correct to reduce the errors (strange shapes can lead to bad segmentation)
            distance      =  ndimage.distance_transform_edt(nucs_bff)                                                   # distance matrix for the watershed
            l_maxi_crd    =  peak_local_max(distance, footprint=np.ones((7, 7)), labels=label(nucs_bff))                # search the local maxima (this is a list of coordinates)
            local_maxi    =  np.zeros((xlen, ylen), dtype=np.uint8)                                                     # local maxima as a matrix, initialization
            for ll in l_maxi_crd:
                local_maxi[ll[0], ll[1]]  =  1                                                                          # put 1 in corrispondence of the peaks
            local_mx      =  gaussian(local_maxi, 1) > 0                                                                # gaussian blurring for the local maxima matrix to avoid to ahve 2 very close peaks
            local_mx_lbl  =  label(local_mx, connectivity=1).astype(np.int32)                                           # label of the local maxima matrix
            local_mx_rgp  =  regionprops_table(local_mx_lbl, properties=["centroid"])                                   # regionprops to extract the centroisd

            ctrs_mx       =  np.zeros((xlen, ylen))                                                                     # initialize centroids matrix
            for cnt, jj in enumerate(local_mx_rgp["centroid-0"]):
                ctrs_mx[np.round(jj).astype(np.int32), np.round(local_mx_rgp['centroid-1'][cnt]).astype(np.int32)]  =  1    # put 1 in corrispondence of the peaks
            markers       =  label(ctrs_mx)                                                                             # markers matrix
            fin_bff       =  watershed(-distance, markers, mask=np.sign(nucs2work[hh]))                                 # finally watershed

            nucs_lbld[hh]  =  label(nucs_lbld[hh]).astype(np.uint32)                                                    # re-label segmented nuclei to avoid high tag values
            nucs_lbld[hh]  =  nucs_lbld[hh] + ((nucs_lbld[hh].max() + 1) * np.sign(fin_bff) + fin_bff).astype(np.uint32)    # add the just segmented nuclei to the principal nuclei segmented matrix

            rgp_bff2  =  regionprops_table(nucs_lbld[hh], properties=["label", "area"])                                 # regionprops of the principal nuclei segmented matrix
            iis2work  =  np.where(rgp_bff2["area"] < 200)[0]                                                            # search for the nuclei with low area
            # ant = np.zeros(nucs_lbld[hh].shape)
            for jj in iis2work:                                                                                         # for all the small area nuclei (here we implement the idea that a peice of nucleus oversegmented must be joined to the nucleus with which it shares more border)
                bff_img   =  (nucs_lbld[hh] == rgp_bff2["label"][jj]) * rgp_bff2["label"][jj]                           # isolate the calc
                # ant += bff_img
                nuc_bff   =  np.sign(expand_labels(bff_img)) - np.sign(bff_img)                                         # expand the calc and remove it from the expantion in order to have the external border of the calc only
                nuc_bff   =  nuc_bff.astype(np.uint32) * nucs_lbld[hh]                                                  # check the tags in the external contour to find the tags of touching nuclei
                if np.isnan(np.median(nuc_bff)):                                                                        # if the matrix is empty (nucleus does not touch anything) the median filter gives nan and there is nothing to do (those are generally pieces of nuclei touching the border)
                    pass
                if np.median(nuc_bff) - np.fix(np.median(nuc_bff)) != 0:                                                # in case you have the same number of pixels in the external contour for 2 tags, you will have a decimal number as result
                    nuc_bff  =  nuc_bff[0]                                                                              # chose the smaller tag: this is comletely arbitrary, but we don't have anything wiser
                elif np.median(nuc_bff) - np.fix(np.median(nuc_bff)) == 0:                                              # if the median gives an integer value, means that there is a tag that is more present than the other(s)
                    nuc_bff  =  np.median(nuc_bff)                                                                      # we take this tag as reference

                nucs_lbld[hh]  *=  (1 - np.sign(bff_img)).astype(np.uint32)                                             # remove the calc from the principal nuclei segmented matrix
                nucs_lbld[hh]  +=  np.sign(bff_img).astype(np.uint32) * np.uint32(nuc_bff)                              # add the small calc to the matrix with the proper tag

        pbar.close()

        mxxs      =  np.asarray(mxxs)
        z_ref     =  mxxs.mean(0)
        z_ref[0]  =  np.round(z_ref[0])
        z_ref[1]  =  np.round(z_ref[1])

        self.nucs_lbld   =  nucs_lbld
        self.green_minp  =  green_minp
        self.z_ref       =  z_ref.astype(int)




        # mycmap  =  np.fromfile("mycmap.bin", "uint16").reshape((10000, 3))   # / 255.0
        # colors4map  =  []
        # for k in range(mycmap.shape[0]):
        #     colors4map.append(mycmap[k, :])
        # colors4map[0]  =  np.array([0, 0, 0])
        #
        # mycmap = pg.ColorMap(np.linspace(0, 1, fin_lbls.max()), color=colors4map)
        # w = pg.image(fin_lbls)
        # # w = pg.image(nucs_lbld)
        # w.setColorMap(mycmap)
