"""This function measures the intensity of GFP around the spots.

It starts from an already done analysis and raw data, frame by frame
measure intesity of gfp channel around each spot.
Input values are 4D raw data and analysis folder.

OSS: HERE WE CALL GREEN THE RFP AND RED THE GFP.

"""

import numpy as np
import datetime
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table
import xlsxwriter
from scipy.optimize import curve_fit

import AnalysisLoader
import UsefulWidgets
import SaveReadMatrix


def exp_func(x, A, alpha):
    """Eponential function."""
    yy  =  A * np.exp(-x / alpha)
    return yy


class CheckIntensityAroundSpots:
    """Measure the intensity of the GFP around the spots."""
    def __init__(self, analysis_folder, fnames, software_version):
    # def __init__(self, analysis_folder, green4d):

        spts_coords             =  np.load(analysis_folder + '/spots_3d_coords.npy')                                    # load the spots 4D coordinate matrix
        tlen, zlen, xlen, ylen  =  spts_coords[-1]                                                                      # extract from the coordinates the shape of the 4d data
        spts_coords             =  spts_coords[:-1]                                                                     # remove this info and select only the spots 4d coordinates
        raw_data                =  AnalysisLoader.RawDataLoader(analysis_folder, fnames)                              # reload raw data files
        # green4d                 =  AnalysisLoader.RawDataLoader(analysis_folder, fnames).green4d                        # reload raw data files
        green4d                 =  raw_data.green4d
        red4d                   =  raw_data.red4d
        nucs_trck               =  np.load(analysis_folder + '/nuclei_tracked.npy')                                     # load tracked nulcei (mip, 2d and time)
        # spts_trck               =  np.load(analysis_folder + '/spots_tracked.npy')                                      # load tracked spots (mip, 2d and time)
        spts_trck               =  SaveReadMatrix.SpotsMatrixReader2D(analysis_folder, '/spots_trck.npy').spts_lbls
        nucs_tags               =  np.unique(nucs_trck[nucs_trck != 0])                                                 # list of all the nuclei's tags

        pbar  =  UsefulWidgets.ProgressBar(total1=tlen)
        pbar.show()
        pbar.update_progressbar(0)

        gfp_spts_mtx  =  np.zeros((nucs_tags.size, tlen))                                                               # initialize the matrix with the info about the intensity grteen channel around the spots
        av_ints_sign  =  np.zeros((tlen))                                                                               # initialize the matrix with the average (per frame) signal intensity
        av_ints_bckg  =  np.zeros((tlen))                                                                               # initialize the matrix with the average (per frame) background intensity
        for tt in range(tlen):                                                                                          # for each time frame
            pbar.update_progressbar(tt)
            spts_singframe    =  np.zeros((zlen, xlen, ylen), dtype=np.uint16)                                          # initialize single 3d frame (z, x, y)
            sub_idxs          =  np.where(spts_coords[:, 0] == tt)[0]                                                   # get position of the coordiantees of the current time frame
            sub_coords        =  spts_coords[sub_idxs]                                                                  # sub matrix with the actual spots coordinates of the current time frame
            spts_singframe[sub_coords[:, 1], sub_coords[:, 2], sub_coords[:, 3]]  =  1                                  # construct the 3d time frame with binary spots
            raw_spts          =  spts_singframe * green4d[tt]                                                           # multiply by raw data to isolate the pixels with the signal
            raw_spts          =  raw_spts[raw_spts != 0]                                                                # remove 0-value pixels
            av_ints_sign[tt]  =  raw_spts.mean()                                                                        # add its mean to the output matrix
            raw_bckg          =  (1 - spts_singframe) * green4d[tt]                                                     # take the negative of the spots calc (the background, the non-signal) and multiply by raw data
            raw_bckg          =  raw_bckg[raw_bckg != 0]                                                                # remove 0-value pixels
            av_ints_bckg[tt]  =  raw_bckg.mean()                                                                        # add its mean to the output matrix
            spts_singframe   *=  spts_trck[tt]                                                                          # multiply by the single frame tracked spots to give the proper labels to the binary spots
            cages             =  expand_labels(spts_singframe, distance=8) - expand_labels(spts_singframe, distance=6)  # define cages around spots using label expantion (you do it with a 6 distance espantion minus a 2 distance expantion)
            cages_rgp         =  regionprops_table(cages, red4d[tt], properties=["label", "intensity_image", "area"])   # regionprops of the cages
            for cnt, uu in enumerate(cages_rgp["label"]):                                                               # for each label in this time frame
                tag_idx                    =  np.where(nucs_tags == uu)[0][0]                                           # search the coordinate of the current tag in the output matrix
                gfp_spts_mtx[tag_idx, tt]  =  np.sum(cages_rgp["intensity_image"][cnt]) / cages_rgp["area"][cnt]        # add the average value of the intensity in the cage

        pbar.close()
        # self.gfp_spts_mtx  =  gfp_spts_mtx
        # self.nucs_tags     =  nucs_tags

        workbook  =  xlsxwriter.Workbook(analysis_folder + "/GfpIntensityAroundSpots.xlsx", {'nan_inf_to_errors': True})
        sheet1    =  workbook.add_worksheet("Info")
        sheet2    =  workbook.add_worksheet("Gfp Intensity Around Spots")
        sheet3    =  workbook.add_worksheet("PhotoBleaching Study")

        sheet1.write(0, 0, "Date")
        sheet1.write(0, 1, datetime.date.today().strftime("%d%b%Y"))
        sheet1.write(2, 0, "Software Version")
        sheet1.write(2, 1, software_version)
        sheet1.write(3, 0, "Analysis_folder")
        sheet1.write(3, 1, analysis_folder)
        sheet1.write(4, 0, "Raw data files")
        for oo, fname in enumerate(fnames):
            sheet1.write(4, 1 + oo, fname)

        sheet2.write(0, 0, "Time")
        for ee in range(tlen):
            sheet2.write(ee + 1, 0, ee)

        for jj, ct in enumerate(nucs_tags):
            sheet2.write(0, jj + 1, "Nuc_" + str(ct))

        for tm in range(tlen):
            for mm in range(nucs_tags.size):
                sheet2.write(1 + tm, 1 + mm, gfp_spts_mtx[mm, tm])

        sheet3.write(0, 0, "Time")
        sheet3.write(0, 1, "Average Signal")
        sheet3.write(0, 2, "Average Background")
        sheet3.write(0, 3, "Average Background Fit")

        popt, pcov  =  curve_fit(exp_func, np.arange(tlen), av_ints_bckg)
        y_fit       =  exp_func(np.arange(tlen), *popt)
        y_fit      *=  100 / y_fit.max()

        for bb in range(tlen):
            sheet3.write(bb + 1, 0, bb)

        for cnt1, qq in enumerate(av_ints_sign):
            sheet3.write(cnt1 + 1, 1, qq)

        for cnt2, ww in enumerate(av_ints_bckg):
            sheet3.write(cnt2 + 1, 2, ww)
            sheet3.write(cnt2 + 1, 3, y_fit[cnt2])

        workbook.close()


# spts_book = load_workbook(analysis_folder + '/journal.xlsx')
# # spts_sheet  =  spts_book.get_sheet_by_name('Ints')
# spts_sheet = spts_book.get_sheet_by_name('Ints by Background')
#
#
# tags = []
# for pp in range(1, gfp_sheet.max_column):
#     tags.append(gfp_sheet.cell(row=1, column=pp + 1).value[4:])
# tags = np.asarray(tags)
#
# for tt in range(1, gfp_sheet.max_row):
#     for k in range(1, gfp_sheet.max_column):
#         spts_gfp_mtx[0, tt - 1, k - 1] = spts_sheet.cell(row=tt + 1, column=k + 3).value
#
# oo = np.where(spts_gfp_mtx[0].sum(0) == 0)[0]
# spts_gfp_mtx = np.delete(spts_gfp_mtx, oo, axis=2)
# tags = np.delete(tags, oo, axis=0)
