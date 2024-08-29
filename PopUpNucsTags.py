"""This class writes .png of single frame nuclei adding nuclei tags as text.

Input is the nuclei frame and the folder to save the file.
"""


import numpy as np
from skimage.measure import regionprops_table
import pyqtgraph.exporters
import pyqtgraph as pg


class PopUpNucsTags:
    """Only class, does all the job."""
    def __init__(self, nucs, analysis_folder, frame_numb):

        mycmap      =  np.fromfile("mycmap.bin", "uint16").reshape((10000, 3))  # / 255.0
        colors4map  =  []
        for k in range(mycmap.shape[0]):
            colors4map.append(mycmap[k, :])
        colors4map[0]  =  np.array([0, 0, 0])

        w       =  pg.image(nucs)
        w.setFixedSize(2000, 2000)
        mycmap  =  pg.ColorMap(np.linspace(0, 1, nucs.max()), color=colors4map)
        w.setColorMap(mycmap)
        tags       =  list()
        rgp_nucs   =  regionprops_table(nucs, properties=["label", "centroid"])
        for jj in rgp_nucs["label"]:
            tags.append(pg.TextItem(str(jj)))
            w.addItem(tags[-1])
        for cntr, dd in enumerate(tags):
            dd.setPos(rgp_nucs["centroid-0"][cntr], rgp_nucs["centroid-1"][cntr])
        exporter                         =  pg.exporters.ImageExporter(w.imageItem)
        exporter.parameters()['width']   =  1000
        exporter.parameters()['height']  =  1000
        exporter.export(analysis_folder + '/nucs_frame' + str(frame_numb) + '.png')







# def sorted_alphanumeric(data):
#     """Function to sort in aplha numeric order"""
#     convert       =  lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key  =  lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
#     return sorted(data, key=alphanum_key)
#
#
# class WriteMultiTiffText:
#     """Only class, does all the job"""
#     def __init__(self, psc_clrs, psc):
#
#         os.mkdir('buffer_tif')
#
#         rgp_cells  =  regionprops_table(psc[0], properties=["label", "centroid"])               # info to write tags on the image
#         zlen       =  psc_clrs.shape[0]
#         # w          =  pg.ImageView()
#         w          =  pg.image(psc_clrs[0])
#         # w.setImage(psc_clrs[0])
#         w.autoRange()
#         tags       =  list()
#         for jj in rgp_cells["label"]:
#             tags.append(pg.TextItem(str(jj)))
#             w.addItem(tags[-1])
#         for cntr, dd in enumerate(tags):
#             dd.setPos(rgp_cells["centroid-0"][cntr], rgp_cells["centroid-1"][cntr])
#         exporter                         =  pg.exporters.ImageExporter(w.imageItem)
#         exporter.parameters()['width']   =  1000
#         exporter.parameters()['height']  =  1000
#         exporter.export('buffer_tif/fileName0.png')
#
#         # pbar  =  ProgressBar(total=zlen)
#         # pbar.update_progressbar(0)
#         # pbar.show()
#
#         for k in range(zlen):
#             # pbar.update_progressbar(k)
#             for tag in tags:
#                 w.removeItem(tag)
#             rgp_cells  =  regionprops_table(psc[k], properties=["label", "centroid"])
#             tags       =  list()
#             for jj in rgp_cells["label"]:
#                 tags.append(pg.TextItem(str(jj)))
#                 w.addItem(tags[-1])
#             for cntr, dd in enumerate(tags):
#                 dd.setPos(rgp_cells["centroid-0"][cntr] - 20, rgp_cells["centroid-1"][cntr] - 20)
#             w.setImage(psc_clrs[k])
#             w.autoRange()
#
#             exporter  =  pg.exporters.ImageExporter(w.imageItem)
#             exporter.parameters()['width']   =  1000
#             exporter.parameters()['height']  =  1000
#             exporter.export('buffer_tif/fileName' + str(k) + '.png')
#
#         a       =  sorted_alphanumeric(os.listdir('buffer_tif'))
#         imlist  =  list()
#         for m in a:
#             imlist.append(Image.open('buffer_tif/' + m))
#
#         imlist[0].save(file2write, save_all=True, compression="tiff_deflate", append_images=imlist[1:])
#
#         shutil.rmtree('buffer_tif/', ignore_errors=True)
#         # pbar.close()
#         w.close()


# class ProgressBar(QtWidgets.QWidget):
#     """Simple progress bar widget"""
#     def __init__(self, parent=None, total=20):
#         super().__init__(parent)
#         self.name_line1  =  QtWidgets.QLineEdit()
#
#         self.progressbar  =  QtWidgets.QProgressBar()
#         self.progressbar.setMinimum(1)
#         self.progressbar.setMaximum(total)
#
#         main_layout  =  QtWidgets.QGridLayout()
#         main_layout.addWidget(self.progressbar, 0, 0)
#
#         self.setLayout(main_layout)
#         self.setWindowTitle("Progress")
#         self.setGeometry(500, 300, 300, 50)
#
#     def update_progressbar(self, val1):
#         """Progress bar updater"""
#         self.progressbar.setValue(val1)
#         QtWidgets.qApp.processEvents()
