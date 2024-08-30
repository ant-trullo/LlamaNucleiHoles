"""This function saves the results of the analysis.

Input are the folder name and all the matrices geenrated. No output.
"""

import os
import datetime
import numpy as np
import xlsxwriter
import tifffile
from skimage.measure import regionprops_table
import pyqtgraph as pg

import SaveReadMatrix

class AnalysisSaver:
    """Only class, does all the job."""
    def __init__(self, folder2write, raw_data, nucs_spots_channels, spots_3d, spots_tracked, nuclei_tracked, nuclei_segmented, features_3d, nuc_active,
                 cages_tli, fnames, gauss_kernelsize_value, spots_thr_value, volume_thr_value, dist_thr_value, max_dist, soft_version):

        os.mkdir(folder2write)
        tifffile.imwrite(str(folder2write) + "/false_2colors.tiff", nuc_active.nuclei_active3c.astype("uint16"))
        np.save(folder2write + '/nucs_spots_channels.npy', nucs_spots_channels)
        np.save(folder2write + '/spots_3d_tzxy.npy', spots_3d.spots_tzxy.astype("uint16"))
        np.save(folder2write + '/spots_3d_coords.npy', spots_3d.spots_coords.astype("uint16"))
        SaveReadMatrix.SpotsMatrixSaver(spots_3d.spots_vol, folder2write, '/spots_3d_vol.npy')
        SaveReadMatrix.SpotsMatrixSaver(spots_3d.spots_ints, folder2write, '/spots_3d_ints.npy')
        SaveReadMatrix.SpotsMatrixSaver2D(spots_tracked, folder2write, '/spots_trck.npy')
        np.save(folder2write + '/nuclei_segmented.npy', nuclei_segmented.nucs_lbld.astype("uint16"))
        np.save(folder2write + '/nuclei_tracked.npy', nuclei_tracked.astype("uint16"))
        np.save(folder2write + '/cages_tli.npy', cages_tli)

        im_red_smpl     =  np.zeros(((2,) + raw_data.imarray_red.shape[1:]), dtype=raw_data.imarray_red.dtype)
        im_red_smpl[0]  =  raw_data.imarray_red[0]
        im_red_smpl[1]  =  raw_data.imarray_red[-1]
        np.save(folder2write + '/im_red_smpl.npy', im_red_smpl)
        np.save(folder2write + '/spots_features3d.npy', features_3d.statistics_info.astype(float))

        idx         =  np.unique(nuclei_tracked[nuclei_tracked != 0])
        spots_ints  =  np.zeros((idx.size, raw_data.imarray_green.shape[0]))
        spots_vols  =  np.zeros((idx.size, raw_data.imarray_green.shape[0]))
        for cnt, k in enumerate(idx):
            spt_bff             =  (spots_tracked == k)
            spots_ints[cnt, :]  =  (spt_bff * spots_3d.spots_ints).sum(2).sum(1)
            spots_vols[cnt, :]  =  (spt_bff * spots_3d.spots_vol).sum(2).sum(1)

        ctrs  =  np.zeros((idx.size, spots_tracked.shape[0], 2))
        i     =  0
        for k in idx:
            aa  =  (spots_tracked == k) * 1
            for t in range(spots_tracked.shape[0]):
                if aa[t, :, :].sum():
                    aa_rgp         =  regionprops_table(aa[t, :, :], properties=["centroid"])
                    ctrs[i, t, :]  =  aa_rgp['centroid-0'][0], aa_rgp['centroid-1'][0]
            i  +=  1

        book    =  xlsxwriter.Workbook(folder2write + '/journal.xlsx')                                                                  # write results
        sheet1  =  book.add_worksheet("Ints")
        sheet2  =  book.add_worksheet("Volume")
        sheet3  =  book.add_worksheet("Background")
        sheet4  =  book.add_worksheet("Ints by Background")
        sheet5  =  book.add_worksheet("Nuclei")
        sheet6  =  book.add_worksheet("Info")

        sheet6.write(0, 0, "Gaussian Kernel size")
        sheet6.write(0, 1, gauss_kernelsize_value)
        sheet6.write(1, 0, "Spots Thr")
        sheet6.write(1, 1, spots_thr_value)
        sheet6.write(2, 0, "Volume Thr")
        sheet6.write(2, 1, volume_thr_value)
        sheet6.write(3, 0, "Distance Thr")
        sheet6.write(3, 1, dist_thr_value)
        sheet6.write(4, 0, "Spot-Nucleus Distance Thr")
        sheet6.write(4, 1, max_dist)
        sheet6.write(5, 0, "software version")
        sheet6.write(5, 1, soft_version)
        sheet6.write(6, 0, "date")
        sheet6.write(6, 1, datetime.datetime.now().strftime("%d-%b-%Y"))
        sheet6.write(7, 0, "files")
        for cnt, ff in enumerate(fnames):
            sheet6.write(8 + cnt, 0, ff[ff.rfind('/') + 1:])

        sheet1.write(0, 0, "Time")
        sheet1.write(0, 1, "Frame")
        sheet1.write(0, 2, "Active Nuc")
        sheet2.write(0, 0, "Time")
        sheet2.write(0, 1, "Frame")
        sheet2.write(0, 2, "Active Nuc")
        sheet3.write(0, 0, "Time")
        sheet3.write(0, 1, "Frame")
        sheet3.write(0, 2, "Active Nuc")
        sheet4.write(0, 0, "Time")
        sheet4.write(0, 1, "Frame")
        sheet4.write(0, 2, "Active Nuc")
        sheet5.write(0, 0, "Time")
        sheet5.write(0, 1, "Frame")

        sheet1.write(2 + int(ctrs.shape[1]), 0, "X coordinate centroid")
        sheet1.write(4 + 2 * int(ctrs.shape[1]), 0, "Y coordinate centroid")

        for t in range(nuc_active.n_active_vector.size):
            sheet1.write(t + 1, 0, t * raw_data.time_step_value)
            sheet1.write(t + 1, 1, t)
            sheet1.write(t + 1, 2, nuc_active.n_active_vector[t])
            sheet2.write(t + 1, 0, t * raw_data.time_step_value)
            sheet2.write(t + 1, 1, t)
            sheet2.write(t + 1, 2, nuc_active.n_active_vector[t])
            sheet3.write(t + 1, 0, t * raw_data.time_step_value)
            sheet3.write(t + 1, 1, t)
            sheet3.write(t + 1, 2, nuc_active.n_active_vector[t])
            sheet4.write(t + 1, 0, t * raw_data.time_step_value)
            sheet4.write(t + 1, 1, t)
            sheet4.write(t + 1, 2, nuc_active.n_active_vector[t])
            sheet5.write(t + 1, 0, t * raw_data.time_step_value)
            sheet5.write(t + 1, 1, t)
            sheet1.write(3 + int(ctrs.shape[1]) + t, 0, t * raw_data.time_step_value)
            sheet1.write(3 + int(ctrs.shape[1]) + t, 1, t)
            sheet1.write(3 + int(ctrs.shape[1]) + t, 2, nuc_active.n_active_vector[t])
            sheet1.write(5 + 2 * int(ctrs.shape[1]) + t, 0, t * raw_data.time_step_value)
            sheet1.write(5 + 2 * int(ctrs.shape[1]) + t, 1, t)
            sheet1.write(5 + 2 * int(ctrs.shape[1]) + t, 2, nuc_active.n_active_vector[t])

        for i in range(spots_ints.shape[0]):
            sheet1.write(0, i + 3, "Spot_" + str(int(idx[i])))
            sheet2.write(0, i + 3, "Spot_" + str(int(idx[i])))
            sheet3.write(0, i + 3, "Spot_" + str(int(idx[i])))
            sheet4.write(0, i + 3, "Spot_" + str(int(idx[i])))
            sheet5.write(0, i + 3, "Nuc_" + str(int(idx[i])))
            for t in range(spots_ints.shape[1]):
                sheet1.write(t + 1, i + 3, spots_ints[i, t])
                sheet2.write(t + 1, i + 3, spots_vols[i, t])
                sheet1.write(t + 3 + int(ctrs.shape[1]), i + 3, ctrs[i, t, 0])
                sheet1.write(t + 5 + 2 * int(ctrs.shape[1]), i + 3, ctrs[i, t, 1])
                cages_subt        =  cages_tli[cages_tli[:, 0] == t]
                cages_sub_sub_tl  =  cages_subt[cages_subt[:, 1] == int(idx[i])]
                if cages_sub_sub_tl.size != 0:
                    sheet3.write(t + 1, i + 3, cages_sub_sub_tl[0, 2])
                    sheet4.write(t + 1, i + 3, spots_ints[i, t] / cages_sub_sub_tl[0, 2])
                elif cages_sub_sub_tl.size == 0:
                    sheet3.write(t + 1, i + 3, 0)
                    sheet4.write(t + 1, i + 3, 0)

        nucs_info  =  np.zeros((nuclei_tracked.shape[0], idx.size))
        for tt in range(nuclei_tracked.shape[0]):
            rgp_nucs  =  regionprops_table(nuclei_tracked[tt], raw_data.red4d[tt, nuclei_segmented.z_ref[0]:nuclei_segmented.z_ref[1]].sum(0), properties=["label", "area", "intensity_image"])
            for cnt, k in enumerate(idx):
                ll  =  np.where(rgp_nucs["label"] == k)[0]
                if ll.size != 0:
                    nucs_info[tt, cnt]  =  np.sum(rgp_nucs["intensity_image"][ll[0]]) / (rgp_nucs["area"][ll[0]] * (nuclei_segmented.z_ref[1] - nuclei_segmented.z_ref[0]))

        for uu in range(nuclei_tracked.shape[0]):
            for oo in range(idx.size):
                sheet5.write(uu + 1, oo + 3, nucs_info[uu, oo])

        book.close()

        x_vv  =  np.arange(nuc_active.n_active_vector.size)
        plt   =  pg.plot()
        plt.plot(x_vv, nuc_active.n_active_vector, pen='r', title='Number of Active Nuclei')
