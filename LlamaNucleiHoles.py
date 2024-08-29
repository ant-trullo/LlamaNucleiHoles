"""This is the main window of the software to analyze Llama data.
This is version 1.0, since May 2023.

author: antonio.trullo@igmm.cnrs.fr
"""

import os
import time
from natsort import natsorted
from importlib import reload
import traceback
import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5 import QtGui, QtWidgets, QtCore

import MultiLoadCzi5D
import NucleiDetector
import LabelsModify
import NucleiConnectMultiCore
import RemoveBadNuclei
import SaveReadMatrix
import SpotsDetectionChopper
import SpotsConnection
import ParametersExtraction
import NucleiSpotsConnection
import PopUpTool
import PopUpNucsTags
import AnalysisSaver
import BackgroundEstimate
import AnalysisLoader
import SpatiallySelectedSaver
import CheckIntensityAroundSpots


class MainWindow(QtWidgets.QMainWindow):
    """Main windows: coordinates all the actions, algorithms, visualization tools and analysis tools."""
    def __init__(self, parent=None):

        QtWidgets.QMainWindow.__init__(self, parent)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        widget  =  QtWidgets.QWidget(self)
        self.setCentralWidget(widget)

        load_data_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/load-hi.png'), "&Load Data", self)
        load_data_action.setShortcut("Ctrl+L")
        load_data_action.setStatusTip("Load raw data files")
        load_data_action.triggered.connect(self.load_raw_data)

        modify_nucs_segm_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/hand-modify.jpg'), "&Modify Nucs", self)
        modify_nucs_segm_action.setShortcut("Ctrl+W")
        modify_nucs_segm_action.setStatusTip("Launch the Manual Modifier Tool")
        modify_nucs_segm_action.triggered.connect(self.modify_nucs_segm)

        chop_stack_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/chop.jpg'), "&chop", self)
        chop_stack_action.setShortcut("Ctrl+M")
        chop_stack_action.setStatusTip("Launch the Manual Modifier Tool")
        chop_stack_action.triggered.connect(self.chop_stack)

        settings_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/settings.png'), "&Settings", self)
        settings_action.setShortcut("Ctrl+T")
        settings_action.setStatusTip("Changes default settings values")
        settings_action.triggered.connect(self.settings_changes)

        save_analysis_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/save-md.png'), "&Save Analysis", self)
        save_analysis_action.setShortcut("Ctrl+S")
        save_analysis_action.setStatusTip("Save analysis")
        save_analysis_action.triggered.connect(self.save_analysis)

        load_analysis_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/load-hi.png'), "&Load Analysis", self)
        load_analysis_action.setShortcut("Ctrl+A")
        load_analysis_action.setStatusTip("Load an already done analysis")
        load_analysis_action.triggered.connect(self.load_analysis)

        exit_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/exit.png'), "&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)

        popup_nuclei_raw_action  =  QtWidgets.QAction("&Pop-up Raw Nuclei", self)
        popup_nuclei_raw_action.setStatusTip("Generate figures with the raw nuclei data")
        popup_nuclei_raw_action.triggered.connect(self.popup_nuclei_raw)

        popup_nuclei_segmented_action  =  QtWidgets.QAction("&Pop-up Segmented Nuclei", self)
        popup_nuclei_segmented_action.setStatusTip("Generate figures with the detected nuclei")
        popup_nuclei_segmented_action.triggered.connect(self.popup_nuclei_segmented)

        popup_nuclei_tracked_action  =  QtWidgets.QAction("&Pop-up Tracked Nuclei", self)
        popup_nuclei_tracked_action.setStatusTip("Generate figures with the tracked nuclei")
        popup_nuclei_tracked_action.triggered.connect(self.popup_nuclei_trackeded)

        popup_spots_raw_action  =  QtWidgets.QAction("&Pop-up Raw Spots", self)
        popup_spots_raw_action.setStatusTip("Generate figures with the raw spots")
        popup_spots_raw_action.triggered.connect(self.popup_spots_raw)

        popup_spots_segm_action  =  QtWidgets.QAction("&Pop-up Segmented Spots", self)
        popup_spots_segm_action.setStatusTip("Generate figures with the segmented spots")
        popup_spots_segm_action.triggered.connect(self.popup_spots_segm)

        popup_nucactive_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/popup.png'), "&Pop-up Active Nuclei", self)
        popup_nucactive_action.setStatusTip("Generate a figure with the active nuclei map")
        popup_nucactive_action.triggered.connect(self.popup_nucactive)

        popup_nucs_with_tags_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/popup.png'), "&Pop-up Tags Nuclei", self)
        popup_nucs_with_tags_action.setStatusTip("Generate a figure with the nuclei and their tags written on the top")
        popup_nucs_with_tags_action.triggered.connect(self.popup_nucs_with_tags)

        spatial_analysis_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/spatial_spots.jpg'), "&Spatial Analysis", self)
        spatial_analysis_action.setShortcut("Ctrl+I")
        spatial_analysis_action.setStatusTip("Split previously done analysis results in spatial domains user defined")
        spatial_analysis_action.triggered.connect(self.spatial_analysis)

        check_gfp_around_spots_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/intensity_around_spots.png'), "&GFP around spots", self)
        check_gfp_around_spots_action.setShortcut("Ctrl+O")
        check_gfp_around_spots_action.setStatusTip("Extimate frame by frame the gfp intensity around spots")
        check_gfp_around_spots_action.triggered.connect(self.check_gfp_around_spots)

        menubar   =  self.menuBar()

        file_menu  =  menubar.addMenu("&File")
        file_menu.addAction(load_data_action)
        file_menu.addAction(save_analysis_action)
        file_menu.addAction(load_analysis_action)
        file_menu.addAction(settings_action)
        file_menu.addAction(exit_action)

        modify  =  menubar.addMenu("&Modify")
        modify.addAction(modify_nucs_segm_action)
        modify.addAction(chop_stack_action)

        popup_menu     =  menubar.addMenu('&PopUp')
        popup_nuclei  =  popup_menu.addMenu(QtGui.QIcon('Icons/popup.png'), "PopUp Nuclei")
        popup_nuclei.addAction(popup_nuclei_raw_action)
        popup_nuclei.addAction(popup_nuclei_segmented_action)
        popup_nuclei.addAction(popup_nuclei_tracked_action)

        popup_spots  =  popup_menu.addMenu(QtGui.QIcon('Icons/popup.png'), "PopUp Spots")
        popup_spots.addAction(popup_spots_raw_action)
        popup_spots.addAction(popup_spots_segm_action)
        popup_menu.addAction(popup_nucactive_action)

        postprocessing  =  menubar.addMenu("&Post-Processing")
        postprocessing.addAction(spatial_analysis_action)
        postprocessing.addAction(check_gfp_around_spots_action)

        fname_raw_edt  =  QtWidgets.QLineEdit("File: ", self)
        fname_raw_edt.setToolTip("Names of the files you are working on")

        frame_nucs_raw  =  pg.ImageView(self, name="NucsRaw")
        frame_nucs_raw.ui.roiBtn.hide()
        frame_nucs_raw.ui.menuBtn.hide()
        frame_nucs_raw.view.setXLink("NucsSgm")
        frame_nucs_raw.view.setYLink("NucsSgm")
        frame_nucs_raw.timeLine.sigPositionChanged.connect(self.update_from_nucs_raw)

        frame_nucs_sgm  =  pg.ImageView(self, name="NucsSgm")
        frame_nucs_sgm.ui.roiBtn.hide()
        frame_nucs_sgm.ui.menuBtn.hide()
        frame_nucs_sgm.view.setXLink("NucsRaw")
        frame_nucs_sgm.view.setYLink("NucsRaw")
        frame_nucs_sgm.timeLine.sigPositionChanged.connect(self.update_from_nucs_sgm)

        frame_spts_raw  =  pg.ImageView(self, name="SpotsRaw")
        frame_spts_raw.ui.roiBtn.hide()
        frame_spts_raw.ui.menuBtn.hide()
        frame_spts_raw.view.setXLink("SpotsSgm")
        frame_spts_raw.view.setYLink("SpotsSgm")
        frame_spts_raw.timeLine.sigPositionChanged.connect(self.update_from_spts_raw)

        frame_spts_sgm  =  pg.ImageView(self, name="SpotsSgm")
        frame_spts_sgm.ui.roiBtn.hide()
        frame_spts_sgm.ui.menuBtn.hide()
        frame_spts_sgm.view.setXLink("SpotsRaw")
        frame_spts_sgm.view.setYLink("SpotsRaw")
        frame_spts_sgm.timeLine.sigPositionChanged.connect(self.update_from_spts_sgm)

        tabs_nucs     =  QtWidgets.QTabWidget()
        tab_nucs_raw  =  QtWidgets.QWidget()
        tab_nucs_sgm  =  QtWidgets.QWidget()

        tabs_spts     =  QtWidgets.QTabWidget()
        tab_spts_raw  =  QtWidgets.QWidget()
        tab_spts_sgm  =  QtWidgets.QWidget()

        frame_nucs_sgm_box  =  QtWidgets.QHBoxLayout()
        frame_nucs_sgm_box.addWidget(frame_nucs_sgm)

        frame_nucs_raw_box  =  QtWidgets.QHBoxLayout()
        frame_nucs_raw_box.addWidget(frame_nucs_raw)

        frame_spts_sgm_box  =  QtWidgets.QHBoxLayout()
        frame_spts_sgm_box.addWidget(frame_spts_sgm)

        frame_spts_raw_box  =  QtWidgets.QHBoxLayout()
        frame_spts_raw_box.addWidget(frame_spts_raw)

        tabs_nucs.addTab(tab_nucs_raw, "Nucs Raw")
        tabs_nucs.addTab(tab_nucs_sgm, "Nucs Segmented")

        tabs_spts.addTab(tab_spts_raw, "Spots Raw")
        tabs_spts.addTab(tab_spts_sgm, "Spots Segmented")

        tab_nucs_raw.setLayout(frame_nucs_raw_box)
        tab_nucs_sgm.setLayout(frame_nucs_sgm_box)

        tab_spts_raw.setLayout(frame_spts_raw_box)
        tab_spts_sgm.setLayout(frame_spts_sgm_box)

        busy_lbl  =  QtWidgets.QLabel("Ready")
        busy_lbl.setStyleSheet("color: green")

        pixsize_x_lbl  =  QtWidgets.QLabel("pix size XY =;")
        pixsize_z_lbl  =  QtWidgets.QLabel("Z step =")

        bottom_box  =  QtWidgets.QHBoxLayout()
        bottom_box.addWidget(busy_lbl)
        bottom_box.addStretch()
        bottom_box.addWidget(pixsize_z_lbl)
        bottom_box.addWidget(pixsize_x_lbl)

        frames_box  =  QtWidgets.QHBoxLayout()
        frames_box.addWidget(tabs_nucs)
        frames_box.addWidget(tabs_spts)

        frames_fnamelbl_box  =  QtWidgets.QVBoxLayout()
        frames_fnamelbl_box.addWidget(fname_raw_edt)
        frames_fnamelbl_box.addLayout(frames_box)
        frames_fnamelbl_box.addLayout(bottom_box)

        segm_nucs_nucs_btn  =  QtWidgets.QPushButton("Segm Nucs", self)
        segm_nucs_nucs_btn.clicked.connect(self.segm_nucs)
        segm_nucs_nucs_btn.setToolTip("Activate the manual tool to modify nuclei")
        segm_nucs_nucs_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        track_nucs_btn  =  QtWidgets.QPushButton("Track", self)
        track_nucs_btn.clicked.connect(self.track_nucs)
        track_nucs_btn.setToolTip("Track Nuclei")
        track_nucs_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        dist_thr_lbl  =  QtWidgets.QLabel('Dist Thr', self)
        dist_thr_lbl.setFixedSize(int(ksf_h * 80), int(ksf_w * 25))

        dist_thr_edt  =  QtWidgets.QLineEdit(self)
        dist_thr_edt.textChanged[str].connect(self.dist_thr_var)
        dist_thr_edt.setToolTip('Distance threshold to track nuclei; suggested value 10')
        dist_thr_edt.setFixedSize(int(ksf_h * 35), int(ksf_w * 25))

        dist_thr_hor  =  QtWidgets.QHBoxLayout()
        dist_thr_hor.addWidget(dist_thr_lbl)
        dist_thr_hor.addWidget(dist_thr_edt)

        spots_detect_btn  =  QtWidgets.QPushButton("S-Detect", self)
        spots_detect_btn.clicked.connect(self.spots_detect)
        spots_detect_btn.setToolTip('Spots Detection')
        spots_detect_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        volume_thr_lbl  =  QtWidgets.QLabel('Vol Thr', self)
        volume_thr_lbl.setFixedSize(int(ksf_h * 80), int(ksf_w * 25))

        volume_thr_edt  =  QtWidgets.QLineEdit(self)
        volume_thr_edt.textChanged[str].connect(self.volume_thr_var)
        volume_thr_edt.setToolTip('Threshold volume on spot detection: suggested value 5')
        volume_thr_edt.setFixedSize(int(ksf_h * 35), int(ksf_w * 25))

        spots_thr_lbl  =  QtWidgets.QLabel('Spots Thr', self)
        spots_thr_lbl.setFixedSize(int(ksf_h * 80), int(ksf_w * 25))

        spots_thr_edt  =  QtWidgets.QLineEdit(self)
        spots_thr_edt.textChanged[str].connect(self.spots_thr_var)
        spots_thr_edt.setToolTip('Intensity threshold to segment spots: it is expressed in terms of standard deviation, suggested value 7')
        spots_thr_edt.setFixedSize(int(ksf_h * 35), int(ksf_w * 25))

        spots_thr_box  =  QtWidgets.QHBoxLayout()
        spots_thr_box.addWidget(spots_thr_lbl)
        spots_thr_box.addWidget(spots_thr_edt)

        volume_thr_box  =  QtWidgets.QHBoxLayout()
        volume_thr_box.addWidget(volume_thr_lbl)
        volume_thr_box.addWidget(volume_thr_edt)

        nuc_spots_conn_btn  =  QtWidgets.QPushButton("S-N Connect", self)
        nuc_spots_conn_btn.clicked.connect(self.nuc_spots_conn)
        nuc_spots_conn_btn.setToolTip('Connect Spots to Nuclei')
        nuc_spots_conn_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        nucs_framenumb_lbl  =  QtWidgets.QLabel('Nucs Frm 0', self)
        nucs_framenumb_lbl.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        spts_framenumb_lbl  =  QtWidgets.QLabel('Spts Frm  0', self)
        spts_framenumb_lbl.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        keys_command_box  =  QtWidgets.QVBoxLayout()
        keys_command_box.addWidget(segm_nucs_nucs_btn)
        keys_command_box.addLayout(dist_thr_hor)
        keys_command_box.addWidget(track_nucs_btn)
        keys_command_box.addLayout(spots_thr_box)
        keys_command_box.addLayout(volume_thr_box)
        keys_command_box.addWidget(spots_detect_btn)
        keys_command_box.addWidget(nuc_spots_conn_btn)
        keys_command_box.addStretch()
        keys_command_box.addStretch()
        keys_command_box.addWidget(nucs_framenumb_lbl)
        keys_command_box.addWidget(spts_framenumb_lbl)
        keys_command_box.addStretch()

        layout  =  QtWidgets.QHBoxLayout(widget)
        layout.addLayout(frames_fnamelbl_box)
        layout.addLayout(keys_command_box)

        mycmap  =  np.fromfile("mycmap.bin", "uint16").reshape((10000, 3)) # / 255.0
        self.colors4map  =  []
        for k in range(mycmap.shape[0]):
            self.colors4map.append(mycmap[k, :])
        self.colors4map[0]  =  np.array([0, 0, 0])

        self.frame_nucs_raw      =  frame_nucs_raw
        self.frame_nucs_sgm      =  frame_nucs_sgm
        self.frame_spts_raw      =  frame_spts_raw
        self.frame_spts_sgm      =  frame_spts_sgm
        self.fname_raw_edt       =  fname_raw_edt
        self.spots_thr_edt       =  spots_thr_edt
        self.volume_thr_edt      =  volume_thr_edt
        self.dist_thr_edt        =  dist_thr_edt
        self.nucs_framenumb_lbl  =  nucs_framenumb_lbl
        self.spts_framenumb_lbl  =  spts_framenumb_lbl
        self.software_version    =  "LlamaNucleiHoles_1.0"
        self.busy_lbl            =  busy_lbl
        self.pixsize_z_lbl       =  pixsize_z_lbl
        self.pixsize_x_lbl       =  pixsize_x_lbl

        self.setGeometry(800, 100, 1200, 800)
        self.setWindowTitle(self.software_version)
        self.setWindowIcon(QtGui.QIcon('Icons/DrosophilaIcon.png'))
        self.show()

    def closeEvent(self, event):
        """Close the GUI, asking confirmation."""
        quit_msg  =  "Are you sure you want to exit the program?"
        reply     =  QtWidgets.QMessageBox.question(self, 'Message', quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def busy_indicator(self):
        """Write a red text (BUSY) as a label on the GUI (bottom left)"""
        self.busy_lbl.setText("Busy")
        self.busy_lbl.setStyleSheet('color: red')

    def ready_indicator(self):
        """Write a green text (READY) as a label on the GUI (bottom left)"""
        self.busy_lbl.setText("Ready")
        self.busy_lbl.setStyleSheet('color: green')

    def update_from_nucs_raw(self):
        """Update nucs segm frame from nucs raw."""
        self.nucs_framenumb_lbl.setText("Nucs Frm " + str(self.frame_nucs_raw.currentIndex))
        try:
            self.frame_nucs_sgm.setCurrentIndex(self.frame_nucs_raw.currentIndex)
        except AttributeError:
            pass

    def update_from_nucs_sgm(self):
        """Update nucs raw frame from nucs sgm."""
        try:
            self.frame_nucs_raw.setCurrentIndex(self.frame_nucs_sgm.currentIndex)
        except AttributeError:
            pass

    def update_from_spts_raw(self):
        """Update spts segm frame from spts raw."""
        self.spts_framenumb_lbl.setText("Spts Frm  " + str(self.frame_spts_raw.currentIndex))
        try:
            self.frame_spts_sgm.setCurrentIndex(self.frame_spts_raw.currentIndex)
        except AttributeError:
            pass

    def update_from_spts_sgm(self):
        """Update spts raw frame from spts sgm."""
        try:
            self.frame_spts_raw.setCurrentIndex(self.frame_spts_sgm.currentIndex)
        except AttributeError:
            pass

    def dist_thr_var(self, text):
        """Set distance threshold value for tracking."""
        self.dist_thr_value  =  float(text)

    def spots_thr_var(self, text):
        """Set threshold value for detection."""
        self.spots_thr_value  =  float(text)

    def volume_thr_var(self, text):
        """Set volume threshold for spots."""
        self.volume_thr_value  =  float(text)

    def chop_stack(self):
        """Call the popup tool to chop the stack."""
        self.mpp2  =  ChopStack(self.raw_data)
        self.mpp2.show()
        self.mpp2.procStart.connect(self.chop_stack_sgnl)

    def chop_stack_sgnl(self):
        """Update main gui with first last frame info."""
        self.raw_data.imarray_red    =  self.raw_data.imarray_red[self.mpp2.first_last_frame[0]:self.mpp2.first_last_frame[1]]
        self.raw_data.imarray_green  =  self.raw_data.imarray_green[self.mpp2.first_last_frame[0]:self.mpp2.first_last_frame[1]]
        self.raw_data.green4d        =  self.raw_data.green4d[self.mpp2.first_last_frame[0]:self.mpp2.first_last_frame[1]]
        self.raw_data.red4d          =  self.raw_data.red4d[self.mpp2.first_last_frame[0]:self.mpp2.first_last_frame[1]]
        self.frame_nucs_raw.setImage(self.raw_data.imarray_red)
        self.frame_spts_raw.setImage(self.raw_data.imarray_green)
        self.mpp2.close()

    def popup_spots_raw(self):
        """Popup green raw data."""
        PopUpTool.PopUpTool(self.raw_data.imarray_green, 'Spots Raw Data')

    def popup_spots_segm(self):
        """Popup tracked spots."""
        PopUpTool.PopUpTool(np.sign(self.spots_tracked_3d), 'Spots Raw Data')

    def popup_nucactive(self):
        """Popup false colored movie of active nuclei."""
        pg.image(self.nuc_active.nuclei_active3c, title="Active Nuclei")
        pg.plot(self.nuc_active.n_active_vector, pen='r', symbol='x')

    def popup_nucs_with_tags(self):
        """Popup false colored movie of active nuclei."""
        analysis_folder  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the folder with the analyzed data"))
        PopUpNucsTags.PopUpNucsTags(self.nuclei_tracked[self.frame_nucs_raw.currentIndex], analysis_folder, self.frame_nucs_sgm.currentIndex)

    def popup_nuclei_raw(self):
        """Popup green raw data."""
        PopUpTool.PopUpTool(self.raw_data.imarray_red, 'Spots Raw Data')

    def popup_nuclei_trackeded(self):
        """Popup green raw data."""
        PopUpTool.PopUpToolWithMap(self.nuclei_tracked, 'Spots Raw Data', self.mycmap)

    def popup_nuclei_segmented(self):
        """Popup green raw data."""
        PopUpTool.PopUpToolWithMap(self.nuclei_segmented.nucs_lbld, 'Spots Raw Data', self.mycmap)

    def load_raw_data(self):
        """Load and concatenate raw data files."""
        reload(MultiLoadCzi5D)
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            self.fnames  =  natsorted(QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi (or lsm) data files to concatenate...", filter="*.lsm *.czi *.tif *.lif")[0])
            # self.fnames  =  ['/home/atrullo/Dropbox/ForAnto_LouiseDistanceProblem/03092023_snallama_sna24MS2_Mother_bcd-GFP_MCP-RFPt_heteroZ_E2_/03092023_snallama_sna24MS2_Mother_bcd-GFP_MCP-RFPt_heteroZ_E2_c.czi']
            joined_fnames = ' '
            for s in self.fnames:
                oo              =  s[s.rfind('/') + 1:]
                joined_fnames  +=  str(oo) + ' ~~~ '
            self.fname_raw_edt.setText(joined_fnames)

            self.nucs_spots_channels  =  SetChannels.getChannels() - 1
            self.raw_data             =  MultiLoadCzi5D.MultiLoadCzi5D(self.fnames, self.nucs_spots_channels)
            # self.first_last_frame     =  [0, self.raw_data.imarray_red.shape[0]]
            self.frame_nucs_raw.setImage(self.raw_data.imarray_red)
            self.frame_spts_raw.setImage(self.raw_data.imarray_green)

            self.pixsize_x_lbl.setText("pix size XY = " + str(self.raw_data.pix_size_x))
            self.pixsize_z_lbl.setText("Z step = " + str(self.raw_data.pix_size_z))

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def segm_nucs(self):
        """Segment nuclei and call the modifier tool."""
        reload(NucleiDetector)
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        # try:
        self.nuclei_segmented  =  NucleiDetector.NucleiDetector(self.raw_data.green4d)

        # except Exception:
        #     traceback.print_exc()

        self.ready_indicator()
        self.mpp1  =  ModifierCycleTool(self.nuclei_segmented.green_minp, self.nuclei_segmented.nucs_lbld, 0)
        self.mpp1.show()
        self.mpp1.procStart.connect(self.sgnl_update_cycle)

    def sgnl_update_cycle(self):
        """Update changes done in the modify manual tool."""
        self.nuclei_segmented.nucs_lbld  =  self.mpp1.nuclei_seg
        self.frame_nucs_sgm.setImage(self.nuclei_segmented.nucs_lbld)
        self.mycmap  =  pg.ColorMap(np.linspace(0, 1, self.nuclei_segmented.nucs_lbld.max()), color=self.colors4map)
        self.frame_nucs_sgm.setColorMap(self.mycmap)
        self.frame_nucs_sgm.setCurrentIndex(self.frame_nucs_raw.currentIndex)
        self.mpp1.close()

    def modify_nucs_segm(self):
        """Launch the ModifierCycleTool."""
        self.mpp1  =  ModifierCycleTool(self.nuclei_segmented.green_minp, self.nuclei_segmented.nucs_lbld, self.frame_nucs_raw.currentIndex)
        self.mpp1.show()
        self.mpp1.procStart.connect(self.sgnl_update_cycle)

    def track_nucs(self):
        """Track segmented nuclei."""
        reload(MultiLoadCzi5D)
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            self.px_brd          =  3
            nuclei_tracked       =  NucleiConnectMultiCore.NucleiConnectMultiCore(self.nuclei_segmented.nucs_lbld, self.dist_thr_value).nuclei_tracked
            self.nuclei_tracked  =  RemoveBadNuclei.RemoveBorderNuclei(nuclei_tracked, self.px_brd).nuclei_tracked
            self.popup_nuclei_trackeded()

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def spots_detect(self):
        """Detect spots in 4D raw data matrix."""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            self.spots_3d         =  SpotsDetectionChopper.SpotsDetectionChopper(self.raw_data.green4d, self.spots_thr_value, self.volume_thr_value)
            self.frame_spts_sgm.setImage(np.sign(self.spots_3d.spots_ints))
            self.frame_spts_sgm.setCurrentIndex(self.frame_spts_raw.currentIndex)

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def nuc_spots_conn(self):
        """Connect detected spots with tracked nuclei."""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            self.max_dist          =  int(InputScalarValue.getNumb(["Numb of pixels", "Max distance in pixels", "Spot Nuc Max Distance"]))
            self.spots_tracked_3d  =  SpotsConnection.SpotsConnection(self.nuclei_tracked, np.sign(self.spots_3d.spots_vol), self.max_dist).spots_tracked
            self.frame_spts_sgm.setImage(np.sign(self.spots_tracked_3d))
            ipp_3d                 =  self.spots_3d.spots_ints.reshape(self.spots_3d.spots_ints.size)
            i                      =  np.where(ipp_3d == 0)[0]
            ipp_3d                 =  np.delete(ipp_3d, i, axis=0)
            self.ipp_3d_av         =  ipp_3d.sum() / float(self.spots_3d.spots_vol.sum())                                                               # ipp is defined just to calculate the average intensity value of the spots, ipp_av
            self.features_3d       =  ParametersExtraction.ParametersExtraction(self.spots_3d.spots_ints, self.spots_tracked_3d, self.spots_3d.spots_vol)   # spots_3D.spots_vol * np.sign(self.spots_tracked_3D))

            self.nuc_active  =  NucleiSpotsConnection.NucleiSpotsConnection(self.spots_tracked_3d, self.nuclei_tracked)
            pg.image(self.nuc_active.nuclei_active3c)
            pg.plot(self.nuc_active.n_active_vector, symbol='x', pen='r')

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def load_analysis(self):
        """Load an already done analysis."""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            analysis_folder  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the folder with the analyzed data"))
            fnames           =  natsorted(QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi (or lsm) data files to concatenate...", filter="*.lsm *.czi *.tif *.lif")[0])
            params           =  AnalysisLoader.AnalysisParameters(analysis_folder)
            self.spots_thr_edt.setText(str(params.spots_thr_value))
            self.volume_thr_edt.setText(str(params.volume_thr_value))
            self.dist_thr_edt.setText(str(params.dist_thr_value))
            self.max_dist    =  params.max_dist

            self.raw_data  =  AnalysisLoader.RawDataLoader(analysis_folder, fnames)
            self.frame_nucs_raw.setImage(self.raw_data.imarray_red)
            self.frame_spts_raw.setImage(self.raw_data.imarray_green)
            self.pixsize_x_lbl.setText("pix size XY = " + str(self.raw_data.pix_size_x))
            self.pixsize_z_lbl.setText("Z step = " + str(self.raw_data.pix_size_z))

            self.nuclei_segmented  =  AnalysisLoader.NucleiSegmented(analysis_folder, self.raw_data.green4d)
            self.frame_nucs_sgm.setImage(self.nuclei_segmented.nucs_lbld)
            self.mycmap            =  pg.ColorMap(np.linspace(0, 1, self.nuclei_segmented.nucs_lbld.max()), color=self.colors4map)
            self.frame_nucs_sgm.setColorMap(self.mycmap)
            self.frame_nucs_sgm.setCurrentIndex(self.frame_nucs_raw.currentIndex)

            self.nuclei_tracked  =  np.load(analysis_folder + '/nuclei_tracked.npy')
            self.popup_nuclei_trackeded()

            self.spots_3d  =  AnalysisLoader.SpotsIntsVol(analysis_folder)
            self.frame_spts_sgm.setImage(np.sign(self.spots_3d.spots_ints))
            self.frame_spts_sgm.setCurrentIndex(self.frame_spts_raw.currentIndex)

            # self.spots_tracked_3d  =  SpotsConnection.SpotsConnection(self.nuclei_tracked, np.sign(self.spots_3d.spots_vol), self.max_dist).spots_tracked
            self.spots_tracked_3d  =  SaveReadMatrix.SpotsMatrixReader2D(analysis_folder, '/spots_trck.npy').spts_lbls

            self.frame_spts_sgm.setImage(np.sign(self.spots_tracked_3d))
            ipp_3d                 =  self.spots_3d.spots_ints.reshape(self.spots_3d.spots_ints.size)
            i                      =  np.where(ipp_3d == 0)[0]
            ipp_3d                 =  np.delete(ipp_3d, i, axis=0)
            self.ipp_3d_av         =  ipp_3d.sum() / float(self.spots_3d.spots_vol.sum())                                                               # ipp is defined just to calculate the average intensity value of the spots, ipp_av
            self.features_3d       =  ParametersExtraction.ParametersExtraction(self.spots_3d.spots_ints, self.spots_tracked_3d, self.spots_3d.spots_vol)   # spots_3D.spots_vol * np.sign(self.spots_tracked_3D))

            self.nuc_active  =  NucleiSpotsConnection.NucleiSpotsConnection(self.spots_tracked_3d, self.nuclei_tracked)
            pg.image(self.nuc_active.nuclei_active3c)
            pg.plot(self.nuc_active.n_active_vector, symbol='x', pen='r')

        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def save_analysis(self):
        """Save analysis."""
        reload(AnalysisSaver)
        folder2write  =  QtWidgets.QFileDialog.getSaveFileName()[0]
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            cages_tli     =  BackgroundEstimate.BackgroundEstimate(self.spots_3d.spots_coords, self.spots_tracked_3d, self.raw_data.green4d)
            AnalysisSaver.AnalysisSaver(folder2write, self.raw_data, self.nucs_spots_channels, self.spots_3d, self.spots_tracked_3d, self.nuclei_tracked, self.nuclei_segmented, self.features_3d, self.nuc_active, cages_tli.cages_tli, self.fnames, self.spots_thr_value, self.volume_thr_value, self.dist_thr_value, self.max_dist, self.software_version)
        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def settings_changes(self):
        """Change settings."""
        self.mpp3  =  SettingsChanges()
        self.mpp3.show()
        self.mpp3.procStart.connect(self.settings_update)

    def settings_update(self):
        """Restart the GUI to make changes in button size effective."""
        self.mpp3.close()
        os.execl(sys.executable, sys.executable, *sys.argv)

    def spatial_analysis(self):
        """Activate tool for spatial analysis post-processing."""
        # fnames           =  ['/home/atrullo/Dropbox/ForAnto_LouiseDistanceProblem/03072023_snallama_sna24MS2_Mother_bcd-GFP_MCP-RFPt_heteroZ_E1/03072023_snallama_sna24MS2_Mother_bcd-GFP_MCP-RFPt_heteroZ_E1_d.czi']
        # analysis_folder  =  '/home/atrullo/Dropbox/Virginia_Anto/LlamaLouise/03092023_snallama_sna24MS2_Mother_bcd-GFP_MCP-RFPt_heteroZ_E2_'
        fnames           =  natsorted(QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi (or lsm) data files to concatenate...", filter="*.lsm *.czi *.tif *.lif")[0])
        analysis_folder  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the folder with the analyzed data"))
        self.mpp4        =  SpatialAnalisys(fnames, analysis_folder)
        self.mpp4.show()

    def check_gfp_around_spots(self):
        """Check intensity of GFP around spots."""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            fnames           =  natsorted(QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi (or lsm) data files to concatenate...", filter="*.lsm *.czi *.tif *.lif")[0])
            analysis_folder  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the folder with the analyzed data"))
            CheckIntensityAroundSpots.CheckIntensityAroundSpots(analysis_folder, fnames, self.software_version)
        except Exception:
            traceback.print_exc()

        self.ready_indicator()


class ModifierCycleTool(QtWidgets.QWidget):
    """Activate the tool to manually correct the segmentation."""
    procStart  =  QtCore.pyqtSignal()
    def __init__(self, imarray_red, nuclei_seg, cif_start):
        QtWidgets.QWidget.__init__(self)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        frameShortcut  =  QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.ShiftModifier + QtCore.Qt.Key_End), self)
        frameShortcut.activated.connect(self.shuffle_clrs)

        self.imarray_red  =  imarray_red
        self.nuclei_seg   =  nuclei_seg
        self.cif_start    =  cif_start

        tabs  =  QtWidgets.QTabWidget()
        tab1  =  QtWidgets.QWidget()
        tab2  =  QtWidgets.QWidget()

        mycmap  =  np.fromfile("mycmap.bin", "uint16").reshape((10000, 3))    # / 255.0
        self.colors4map  =  []
        for k in range(mycmap.shape[0]):
            self.colors4map.append(mycmap[k, :])
        self.colors4map[0]  =  np.array([0, 0, 0])

        framepp1  =  pg.ImageView(self, name='Frame1')
        framepp1.getImageItem().mouseClickEvent  =  self.click
        framepp1.ui.roiBtn.hide()
        framepp1.ui.menuBtn.hide()
        framepp1.setImage(self.nuclei_seg)
        mycmap  =  pg.ColorMap(np.linspace(0, 1, self.nuclei_seg.max()), color=self.colors4map)
        framepp1.setColorMap(mycmap)
        framepp1.timeLine.sigPositionChanged.connect(self.update_frame2)

        framepp2  =  pg.ImageView(self)
        framepp2.ui.roiBtn.hide()
        framepp2.ui.menuBtn.hide()
        framepp2.setImage(self.imarray_red)
        framepp2.timeLine.sigPositionChanged.connect(self.update_frame1)
        framepp2.view.setXLink('Frame1')
        framepp2.view.setYLink('Frame1')

        shuffle_clrs_btn  =  QtWidgets.QPushButton("Shuffle Colors", self)
        shuffle_clrs_btn.setFixedSize(int(ksf_h * 120), int(ksf_w * 25))
        shuffle_clrs_btn.clicked.connect(self.shuffle_clrs)
        shuffle_clrs_btn.setToolTip('Shuffle colors')

        modify_btn  =  QtWidgets.QPushButton("Modify", self)
        modify_btn.setFixedSize(int(ksf_h * 120), int(ksf_w * 25))
        modify_btn.clicked.connect(self.modify_lbls)
        modify_btn.setToolTip('Modify selection accordin to the segment (Ctrl+Suppr)')

        update_mainwindows_btn  =  QtWidgets.QPushButton("Update Nuclei", self)
        update_mainwindows_btn.setFixedSize(int(ksf_h * 120), int(ksf_w * 25))
        update_mainwindows_btn.clicked.connect(self.update_mainwindows)

        frame_numb_lbl = QtWidgets.QLabel("frame  " + '0', self)
        frame_numb_lbl.setFixedSize(int(ksf_h * 110), int(ksf_w * 25))

        end_pts  =  np.zeros((2, 2, 100))
        ar_reg   =  np.zeros((1000, 100))

        frame1_box  =  QtWidgets.QHBoxLayout()
        frame1_box.addWidget(framepp1)

        frame2_box  =  QtWidgets.QHBoxLayout()
        frame2_box.addWidget(framepp2)

        btn_box  =  QtWidgets.QHBoxLayout()
        btn_box.addWidget(shuffle_clrs_btn)
        btn_box.addStretch()
        btn_box.addWidget(frame_numb_lbl)
        btn_box.addWidget(modify_btn)
        btn_box.addWidget(update_mainwindows_btn)

        tab1.setLayout(frame1_box)
        tab2.setLayout(frame2_box)

        tabs.addTab(tab1, "Segmented")
        tabs.addTab(tab2, "RAW")

        layout  =  QtWidgets.QVBoxLayout()
        layout.addWidget(tabs)                                             # tabs is a Widget not a Layout!!!!!
        layout.addLayout(btn_box)

        self.end_pts         =  end_pts
        self.ar_reg          =  ar_reg
        self.framepp1        =  framepp1
        self.framepp2        =  framepp2
        self.frame_numb_lbl  =  frame_numb_lbl
        self.c_count         =  0

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("Modifier Tool")

        self.framepp1.setCurrentIndex(self.cif_start)

    def keyPressEvent(self, event):
        if event.key() == (QtCore.Qt.ControlModifier and Qt.Key_Z):

            cif                                =  self.framepp1.currentIndex
            self.nuclei_seg[cif, :, :]         =  self.bufframe
            self.framepp1.updateImage()
            self.framepp1.setCurrentIndex(cif)

        if event.key() == (QtCore.Qt.ControlModifier and Qt.Key_Delete):
            self.modify_lbls()

    def click(self, event):
        event.accept()
        pos        =  event.pos()
        modifiers  =  QtWidgets.QApplication.keyboardModifiers()

        if modifiers  ==  QtCore.Qt.ShiftModifier:
            if self.c_count - 2 * (self.c_count // 2) == 0:
                self.pos1  =  pos
            else:
                try:
                    self.framepp1.removeItem(self.roi)
                except AttributeError:
                    pass

                self.roi      =  pg.LineSegmentROI([self.pos1, pos], pen='r')
                self.framepp1.addItem(self.roi)

            self.c_count  +=  1

    def modify_lbls(self):

        cif      =  self.framepp1.currentIndex
        pp       =  self.roi.getHandles()
        pp       =  [self.roi.mapToItem(self.framepp1.imageItem, p.pos()) for p in pp]
        end_pts  =  np.array([[int(pp[0].x()), int(pp[0].y())], [int(pp[1].x()), int(pp[1].y())]])
        bufframe                           =  np.copy(self.nuclei_seg[cif, :, :])
        self.nuclei_seg[cif, :, :]         =  LabelsModify.LabelsModify(self.nuclei_seg[cif, :, :], end_pts).labels_fin
        self.framepp1.updateImage()
        self.bufframe  =  bufframe

    def update_frame2(self):
        self.framepp2.setCurrentIndex(self.framepp1.currentIndex)
        self.frame_numb_lbl.setText("frame  "  +  str(self.framepp1.currentIndex))

    def update_frame1(self):
        self.framepp1.setCurrentIndex(self.framepp2.currentIndex)
        self.frame_numb_lbl.setText("frame  "  +  str(self.framepp2.currentIndex))

    def shuffle_clrs(self):
        colors_bff           =  self.colors4map[1:]
        np.random.shuffle(colors_bff)
        self.colors4map[1:]  =  colors_bff
        mycmap               =  pg.ColorMap(np.linspace(0, 1, self.nuclei_seg.max()), color=self.colors4map)
        self.framepp1.setColorMap(mycmap)
        self.framepp1.updateImage()

    @QtCore.pyqtSlot()
    def update_mainwindows(self):
        """Send result to main window."""
        self.procStart.emit()

class ChopStack(QtWidgets.QWidget):
    """Popup tool to remove frames from the raw data stack."""
    procStart  =  QtCore.pyqtSignal()

    def __init__(self, raw_data):
        QtWidgets.QWidget.__init__(self)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        frame_red  =  pg.ImageView(self, name='FrameSpts1')
        frame_red.ui.roiBtn.hide()
        frame_red.ui.menuBtn.hide()
        frame_red.timeLine.sigPositionChanged.connect(self.update_frame_from_red)
        frame_red.setImage(raw_data.imarray_red)

        frame_green  =  pg.ImageView(self)
        frame_green.ui.roiBtn.hide()
        frame_green.ui.menuBtn.hide()
        frame_green.view.setXLink('FrameSpts1')
        frame_green.view.setYLink('FrameSpts1')
        frame_green.timeLine.sigPositionChanged.connect(self.update_frame_from_green)
        frame_green.setImage(raw_data.imarray_green)

        tabs_tot   =  QtWidgets.QTabWidget()
        tab_red    =  QtWidgets.QWidget()
        tab_green  =  QtWidgets.QWidget()

        frame_box_red  =  QtWidgets.QHBoxLayout()
        frame_box_red.addWidget(frame_red)

        frame_box_green  =  QtWidgets.QHBoxLayout()
        frame_box_green.addWidget(frame_green)

        tab_red.setLayout(frame_box_red)
        tab_green.setLayout(frame_box_green)

        tabs_tot.addTab(tab_red, "Nucs")
        tabs_tot.addTab(tab_green, "Spots")

        current_frame_lbl  =  QtWidgets.QLabel("Current Frame 0", self)
        current_frame_lbl.setFixedSize(int(ksf_h * 250), int(ksf_w * 25))

        set_first_frame_btn  =  QtWidgets.QPushButton("Set First", self)
        set_first_frame_btn.clicked.connect(self.set_first_frame)
        set_first_frame_btn.setToolTip('Set current frame as first analysis frame')
        set_first_frame_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        set_last_frame_btn  =  QtWidgets.QPushButton("Set Last", self)
        set_last_frame_btn.clicked.connect(self.set_last_frame)
        set_last_frame_btn.setToolTip('Set current frame as last analysis frame')
        set_last_frame_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        send_btn  =  QtWidgets.QPushButton("Send", self)
        send_btn.clicked.connect(self.close_insert)
        send_btn.setToolTip('Remove selected frames and apply changes in main GUI')
        send_btn.setFixedSize(int(ksf_h * 180), int(ksf_w * 25))

        lbls_buttons_box  =  QtWidgets.QHBoxLayout()
        lbls_buttons_box.addWidget(current_frame_lbl)
        lbls_buttons_box.addStretch()
        lbls_buttons_box.addWidget(set_first_frame_btn)
        lbls_buttons_box.addWidget(set_last_frame_btn)
        lbls_buttons_box.addWidget(send_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addWidget(tabs_tot)
        layout.addLayout(lbls_buttons_box)

        self.frame_red            =  frame_red
        self.frame_green          =  frame_green
        self.current_frame_lbl    =  current_frame_lbl
        self.set_first_frame_btn  =  set_first_frame_btn
        self.set_last_frame_btn   =  set_last_frame_btn
        self.first_last_frame     =  np.array([0, raw_data.imarray_red.shape[0]])

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("ChopStack Tool")

    def update_frame_from_red(self):
        """Update frame index from red to green.."""
        self.frame_green.setCurrentIndex(self.frame_red.currentIndex)
        self.current_frame_lbl.setText("Current frame " + str(self.frame_green.currentIndex))

    def update_frame_from_green(self):
        """Update frame index from green to red."""
        self.frame_red.setCurrentIndex(self.frame_green.currentIndex)

    def set_first_frame(self):
        """Set first frame choise."""
        self.first_last_frame[0]  =  self.frame_red.currentIndex
        self.set_first_frame_btn.setText("First " + str(self.frame_green.currentIndex))
        # self.first_frame_lbl.setText("First Frame " + str(self.first_last_frame[0]))

    def set_last_frame(self):
        """Set last frame choise."""
        self.first_last_frame[1]  =  self.frame_red.currentIndex + 1     # we want last frame to be included
        self.set_last_frame_btn.setText("Last " + str(self.frame_green.currentIndex))
        # self.last_frame_lbl.setText("Last Frame " + str(self.first_last_frame[1]))

    @QtCore.pyqtSlot()
    def close_insert(self):
        """Send message to main GUI."""
        self.procStart.emit()


class SpatialAnalisys(QtWidgets.QWidget):
    """Popup tool to study spots detection."""
    def __init__(self, fnames, analysis_folder):
        QtWidgets.QWidget.__init__(self)

        raw_data  =  MultiLoadCzi5D.MultiLoadCzi5D(fnames, np.load(analysis_folder +  '/nucs_spots_channels.npy'))

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        frame1  =  pg.ImageView(self, name="Frame1")
        frame1.setImage(raw_data.imarray_red)
        frame1.ui.roiBtn.hide()
        frame1.ui.menuBtn.hide()
        frame1.view.setXLink("Frame2")
        frame1.view.setYLink("Frame2")
        frame1.timeLine.sigPositionChanged.connect(self.update_from_frame1)

        frame2  =  pg.ImageView(self, name="Frame2")
        frame2.setImage(raw_data.imarray_green)
        frame2.ui.roiBtn.hide()
        frame2.ui.menuBtn.hide()
        frame2.view.setXLink("Frame1")
        frame2.view.setYLink("Frame1")
        frame2.timeLine.sigPositionChanged.connect(self.update_from_frame2)

        tabs       =  QtWidgets.QTabWidget()
        tab_red    =  QtWidgets.QWidget()
        tab_green  =  QtWidgets.QWidget()

        frame1_box  =  QtWidgets.QHBoxLayout()
        frame1_box.addWidget(frame1)

        frame2_box  =  QtWidgets.QHBoxLayout()
        frame2_box.addWidget(frame2)

        tab_red.setLayout(frame1_box)
        tab_green.setLayout(frame2_box)

        tabs.addTab(tab_red, "Llama")
        tabs.addTab(tab_green, "GFP")

        foldername_edt  =  QtWidgets.QLineEdit(analysis_folder, self)
        # foldername_edt.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        first_analyzed_lbl  =  QtWidgets.QLabel(self)
        last_analyzed_lbl   =  QtWidgets.QLabel(self)
        first_frame         =  np.load(analysis_folder + '/im_red_smpl.npy')[0]
        last_frame          =  np.load(analysis_folder + '/im_red_smpl.npy')[1]

        uu_first  =  np.sum(raw_data.imarray_red - first_frame, axis=(1, 2))
        try:
            first_fr  =  np.where(uu_first == 0)[0][0]
            first_analyzed_lbl.setText("First Analyzed Frm: " + str(first_fr))
        except IndexError:
            first_analyzed_lbl.setText("First Analyzed Frm out")

        uu_last  =  np.sum(raw_data.imarray_red - last_frame, axis=(1, 2))
        try:
            last_fr  =  np.where(uu_last == 0)[0][0]
            last_analyzed_lbl.setText("Last Analyzed Frm: " + str(last_fr))
        except IndexError:
            last_analyzed_lbl.setText("Last Analyzed Frm out")

        foldname_frame_box  =  QtWidgets.QVBoxLayout()
        foldname_frame_box.addWidget(foldername_edt)
        foldname_frame_box.addWidget(tabs)

        time_lbl  =  QtWidgets.QLabel("time 00:00", self)
        time_lbl.setFixedSize(int(ksf_h * 120), int(ksf_w * 25))

        frame_lbl  =  QtWidgets.QLabel("frame 0", self)
        frame_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        title_lbl  =  QtWidgets.QLabel("Roi Coords (µm)", self)
        title_lbl.setFixedSize(int(ksf_h * 160), int(ksf_w * 25))

        roi_left_edt  =  QtWidgets.QLineEdit(self)
        roi_left_edt.textChanged[str].connect(self.roi_left_var)
        roi_left_edt.setToolTip("Sets/Read left corners x-coordinate")
        roi_left_edt.setFixedSize(int(ksf_h * 70), int(ksf_w * 25))
        roi_left_edt.returnPressed.connect(self.roi_left_update)

        left_lbl  =  QtWidgets.QLabel("left ", self)
        left_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        left_box  =  QtWidgets.QVBoxLayout()
        left_box.addWidget(left_lbl)
        left_box.addWidget(roi_left_edt)

        roi_right_edt  =  QtWidgets.QLineEdit(self)
        roi_right_edt.textChanged[str].connect(self.roi_right_var)
        roi_right_edt.setToolTip("Sets/Read right corners x-coordinate")
        roi_right_edt.setFixedSize(int(ksf_h * 70), int(ksf_w * 25))
        roi_right_edt.returnPressed.connect(self.roi_right_update)

        right_lbl  =  QtWidgets.QLabel("right ", self)
        right_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        right_box  =  QtWidgets.QVBoxLayout()
        right_box.addWidget(right_lbl)
        right_box.addWidget(roi_right_edt)

        roi_bottom_edt  =  QtWidgets.QLineEdit(self)
        roi_bottom_edt.textChanged[str].connect(self.roi_bottom_var)
        roi_bottom_edt.setToolTip("Sets/Read bottom corner y-coordinate")
        roi_bottom_edt.setFixedSize(int(ksf_h * 70), int(ksf_w * 25))
        roi_bottom_edt.returnPressed.connect(self.roi_bottom_update)

        bottom_lbl  =  QtWidgets.QLabel("bottom ", self)
        bottom_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        bottom_box  =  QtWidgets.QVBoxLayout()
        bottom_box.addWidget(bottom_lbl)
        bottom_box.addWidget(roi_bottom_edt)

        roi_top_edt  =  QtWidgets.QLineEdit(self)
        roi_top_edt.textChanged[str].connect(self.roi_top_var)
        roi_top_edt.setToolTip("Sets/Read top corners y-coordinate")
        roi_top_edt.setFixedSize(int(ksf_h * 70), int(ksf_w * 25))
        roi_top_edt.returnPressed.connect(self.roi_top_update)

        top_lbl  =  QtWidgets.QLabel("top ", self)
        top_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        top_box  =  QtWidgets.QVBoxLayout()
        top_box.addWidget(top_lbl)
        top_box.addWidget(roi_top_edt)

        roi_coords_box  =  QtWidgets.QGridLayout()
        roi_coords_box.addLayout(left_box, 0, 0)
        roi_coords_box.addLayout(right_box, 0, 1)
        roi_coords_box.addLayout(bottom_box, 1, 0)
        roi_coords_box.addLayout(top_box, 1, 1)

        save_region_btn  =  QtWidgets.QPushButton("Save", self)
        save_region_btn.clicked.connect(self.save_region)
        save_region_btn.setToolTip("Save Excel file with only the spots present in the region of interest")
        save_region_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        commands  =  QtWidgets.QVBoxLayout()
        commands.addWidget(title_lbl)
        commands.addLayout(roi_coords_box)
        commands.addStretch()
        commands.addWidget(first_analyzed_lbl)
        commands.addWidget(last_analyzed_lbl)
        commands.addStretch()
        commands.addWidget(save_region_btn)
        commands.addWidget(time_lbl)
        commands.addWidget(frame_lbl)

        layout  =  QtWidgets.QHBoxLayout()
        layout.addLayout(foldname_frame_box)
        layout.addLayout(commands)

        crop_roi1  =  pg.RectROI([80, 80], [80, 80], pen='r')
        crop_roi1.sigRegionChanged.connect(self.roi_update_coords)

        crop_roi2  =  pg.RectROI([80, 80], [80, 80], pen='r')
        crop_roi2.sigRegionChanged.connect(self.roi_update_roi1)

        frame1.addItem(crop_roi1)
        frame2.addItem(crop_roi2)

        self.frame1     =  frame1
        self.frame2     =  frame2
        self.frame_lbl  =  frame_lbl
        self.time_lbl   =  time_lbl
        self.raw_data   =  raw_data
        self.crop_roi1  =  crop_roi1
        self.crop_roi2  =  crop_roi2

        self.roi_left_edt     =  roi_left_edt
        self.roi_right_edt    =  roi_right_edt
        self.roi_bottom_edt   =  roi_bottom_edt
        self.roi_top_edt      =  roi_top_edt
        self.analysis_folder  =  analysis_folder

        self.setLayout(layout)
        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle("Test Spot Detection")

    def update_from_frame1(self):
        """Update labels and frame2 from frame1 current index."""
        self.frame2.setCurrentIndex(self.frame1.currentIndex)
        self.frame_lbl.setText("frame " + str(self.frame1.currentIndex))
        self.time_lbl.setText("time " + time.strftime("%M:%S", time.gmtime(self.frame1.currentIndex * self.raw_data.time_step_value)))

    def update_from_frame2(self):
        """Update frame1 from frame2 current index."""
        self.frame1.setCurrentIndex(self.frame2.currentIndex)

    def roi_update_coords(self):
        """Update coordinate value of the roi corners coordinate and roi2 from roi1."""
        self.roi_left_edt.setText(str(np.round(self.crop_roi1.getState()["pos"][0] * self.raw_data.pix_size_x, 2)))
        self.roi_right_edt.setText(str(np.round((self.crop_roi1.getState()["pos"][0] + self.crop_roi1.getState()["size"][0]) * self.raw_data.pix_size_x, 2)))
        self.roi_top_edt.setText(str(np.round(self.crop_roi1.getState()["pos"][1] * self.raw_data.pix_size_x, 2)))
        self.roi_bottom_edt.setText(str(np.round((self.crop_roi1.getState()["pos"][1] + self.crop_roi1.getState()["size"][1]) * self.raw_data.pix_size_x, 2)))
        self.crop_roi2.sigRegionChanged.disconnect(self.roi_update_roi1)
        self.crop_roi2.setState(self.crop_roi1.getState())
        self.crop_roi2.sigRegionChanged.connect(self.roi_update_roi1)

    def roi_update_roi1(self):
        """Update roi1 from roi2."""
        self.crop_roi1.setState(self.crop_roi2.getState())

    def roi_top_var(self, text):
        """Set roi top y-coordinates value."""
        self.roi_top_value  =  float(text)

    def roi_bottom_var(self, text):
        """Set roi bottom y-coordinates value."""
        self.roi_bottom_value  =  float(text)

    def roi_right_var(self, text):
        """Set roi right x-coordinates value."""
        self.roi_right_value  =  float(text)

    def roi_left_var(self, text):
        """Set roi left x-coordinates value."""
        self.roi_left_value  =  float(text)

    def roi_right_update(self):
        """Set the right corners x-coordinate."""
        self.crop_roi1.sigRegionChanged.disconnect(self.roi_update_coords)
        self.crop_roi1.setSize([(self.roi_right_value / (self.raw_data.pix_size_x) - self.crop_roi1.getState()["pos"][0]), self.crop_roi1.getState()["size"][1]])
        self.crop_roi1.sigRegionChanged.connect(self.roi_update_coords)
        self.crop_roi2.sigRegionChanged.disconnect(self.roi_update_roi1)
        self.crop_roi2.setState(self.crop_roi1.getState())
        self.crop_roi2.sigRegionChanged.connect(self.roi_update_roi1)

    def roi_left_update(self):
        """Set the right corners x-coordinate."""
        self.crop_roi1.sigRegionChanged.disconnect(self.roi_update_coords)
        self.crop_roi1.setPos([self.roi_left_value / (self.raw_data.pix_size_x), self.crop_roi1.getState()["pos"][1]])
        self.crop_roi1.sigRegionChanged.connect(self.roi_update_coords)
        self.roi_right_update()
        self.crop_roi2.sigRegionChanged.disconnect(self.roi_update_roi1)
        self.crop_roi2.setState(self.crop_roi1.getState())
        self.crop_roi2.sigRegionChanged.connect(self.roi_update_roi1)

    def roi_top_update(self):
        """Set the top corners y-coordinate."""
        self.crop_roi1.sigRegionChanged.disconnect(self.roi_update_coords)
        self.crop_roi1.setPos([self.crop_roi1.getState()["pos"][0], self.roi_top_value / (self.raw_data.pix_size_x)])
        self.crop_roi1.sigRegionChanged.connect(self.roi_update_coords)
        self.roi_bottom_update()
        self.crop_roi2.sigRegionChanged.disconnect(self.roi_update_roi1)
        self.crop_roi2.setState(self.crop_roi1.getState())
        self.crop_roi2.sigRegionChanged.connect(self.roi_update_roi1)

    def roi_bottom_update(self):
        """Set the bottom corners y-coordinate."""
        self.crop_roi1.sigRegionChanged.disconnect(self.roi_update_coords)
        self.crop_roi1.setSize([self.crop_roi1.getState()["size"][0], self.roi_bottom_value / (self.raw_data.pix_size_x) - self.crop_roi1.getState()["pos"][1]])
        self.crop_roi1.sigRegionChanged.connect(self.roi_update_coords)
        self.crop_roi2.sigRegionChanged.disconnect(self.roi_update_roi1)
        self.crop_roi2.setState(self.crop_roi1.getState())
        self.crop_roi2.sigRegionChanged.connect(self.roi_update_roi1)

    def save_region(self):
        """Save the spatially organized Excell file."""
        reload(SpatiallySelectedSaver)
        xlsx_filename  =  str(QtWidgets.QFileDialog.getSaveFileName(None, "Define the excell file to write spatially selected traces", filter="*.xlsx")[0])
        left           =  min(self.roi_left_value / (self.raw_data.pix_size_x), self.roi_right_value / (self.raw_data.pix_size_x))
        right          =  max(self.roi_left_value / (self.raw_data.pix_size_x), self.roi_right_value / (self.raw_data.pix_size_x))
        top            =  min(self.roi_top_value / (self.raw_data.pix_size_x), self.roi_bottom_value / (self.raw_data.pix_size_x))
        bottom         =  max(self.roi_top_value / (self.raw_data.pix_size_x), self.roi_bottom_value / (self.raw_data.pix_size_x))
        SpatiallySelectedSaver.SpatiallySelectedSaver(xlsx_filename, self.analysis_folder, left, right, top, bottom, self.raw_data.pix_size_x)



class InputScalarValue(QtWidgets.QDialog):
    """Popup dialog to input a number"""
    def __init__(self, texts, parent=None):
        super().__init__(parent)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        numb_pixels_lbl  =  QtWidgets.QLabel(texts[0], self)
        numb_pixels_lbl.setFixedSize(int(ksf_h * 110), int(ksf_w * 25))

        numb_pixels_edt = QtWidgets.QLineEdit(self)
        numb_pixels_edt.setToolTip(texts[1])
        numb_pixels_edt.setFixedSize(int(ksf_h * 100), int(ksf_w * 22))
        numb_pixels_edt.textChanged[str].connect(self.numb_pixels_var)

        input_close_btn  =  QtWidgets.QPushButton("Ok", self)
        input_close_btn.clicked.connect(self.input_close)
        input_close_btn.setToolTip('Input values')
        input_close_btn.setFixedSize(int(ksf_h * 50), int(ksf_w * 25))

        numb_pixels_lbl_edit_box  =  QtWidgets.QHBoxLayout()
        numb_pixels_lbl_edit_box.addWidget(numb_pixels_lbl)
        numb_pixels_lbl_edit_box.addWidget(numb_pixels_edt)

        input_close_box  =  QtWidgets.QHBoxLayout()
        input_close_box.addStretch()
        input_close_box.addWidget(input_close_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addLayout(numb_pixels_lbl_edit_box)
        layout.addLayout(input_close_box)

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, 160, 120)
        self.setWindowTitle(texts[2])

    def numb_pixels_var(self, text):
        """Set the value ('numb_pixels' is just random name)."""
        self.numb_pixels_value  =  float(text)

    def input_close(self):
        """Input value and quit."""
        self.close()

    def numb_pixels(self):
        """Fucntion to return the value."""
        return self.numb_pixels_value

    @staticmethod
    def getNumb(parent=None):
        dialog       =  InputScalarValue(parent)
        result       =  dialog.exec_()
        numb_pixels  =  dialog.numb_pixels()
        return numb_pixels


class SettingsChanges(QtWidgets.QWidget):
    """Tool to change visualization and analysis parameters."""
    procStart  =  QtCore.pyqtSignal()

    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        self.ksf_h                       =  np.load('keys_size_factor.npy')[0]
        self.ksf_w                       =  np.load('keys_size_factor.npy')[1]
        self.spts_nucs_default_channels  =  np.load('nucs_spts_default_channels.npy')

        ksf_h_lbl  =  QtWidgets.QLabel("Keys Scale Factor W")

        ksf_h_edt  =  QtWidgets.QLineEdit(self)
        ksf_h_edt.textChanged[str].connect(self.ksf_h_var)
        ksf_h_edt.setToolTip("Sets keys scale size (width)")
        ksf_h_edt.setFixedSize(int(self.ksf_h * 65), int(self.ksf_w * 25))
        ksf_h_edt.setText(str(self.ksf_h))

        ksf_w_lbl  =  QtWidgets.QLabel("Keys Scale Factor H")

        ksf_w_edt  =  QtWidgets.QLineEdit(self)
        ksf_w_edt.textChanged[str].connect(self.ksf_w_var)
        ksf_w_edt.setToolTip("Sets keys scale size (heigth)")
        ksf_w_edt.setFixedSize(int(self.ksf_h * 65), int(self.ksf_w * 25))
        ksf_w_edt.setText(str(self.ksf_w))

        nuclei_channel_combo  =  QtWidgets.QComboBox(self)
        nuclei_channel_combo.addItem("None")
        nuclei_channel_combo.addItem("1")
        nuclei_channel_combo.addItem("2")
        nuclei_channel_combo.activated[str].connect(self.nuclei_channel_switch)
        nuclei_channel_combo.setCurrentIndex(self.spts_nucs_default_channels[0])
        nuclei_channel_combo.setFixedSize(int(self.ksf_h * 65), int(self.ksf_w * 25))
        self.nuclei_channel_switch(nuclei_channel_combo.currentText())

        spots_channel_combo  =  QtWidgets.QComboBox(self)
        spots_channel_combo.addItem("None")
        spots_channel_combo.addItem("1")
        spots_channel_combo.addItem("2")
        spots_channel_combo.activated[str].connect(self.spots_channel_switch)
        spots_channel_combo.setCurrentIndex(self.spts_nucs_default_channels[1])
        spots_channel_combo.setFixedSize(int(self.ksf_h * 65), int(self.ksf_w * 25))
        self.spots_channel_switch(nuclei_channel_combo.currentText())

        nuclei_channel_lbl  =  QtWidgets.QLabel("Nuclei Channel", self)

        spots_channel_lbl  =  QtWidgets.QLabel("Spots Channel", self)

        save_btn  =  QtWidgets.QPushButton("Save", self)
        save_btn.clicked.connect(self.save_vars)
        save_btn.setToolTip('Make default the choseen parameters')
        save_btn.setFixedSize(int(self.ksf_h * 50), int(self.ksf_w * 25))

        close_btn  =  QtWidgets.QPushButton("Close", self)
        close_btn.clicked.connect(self.close_)
        close_btn.setToolTip('Close Widget')
        close_btn.setFixedSize(int(self.ksf_h * 50), int(self.ksf_w * 25))

        restart_btn  =  QtWidgets.QPushButton("Refresh", self)
        restart_btn.clicked.connect(self.restart)
        restart_btn.setToolTip('Refresh GUI')
        restart_btn.setFixedSize(int(self.ksf_h * 60), int(self.ksf_w * 25))

        btns_box  =  QtWidgets.QHBoxLayout()
        btns_box.addWidget(save_btn)
        btns_box.addWidget(restart_btn)
        btns_box.addWidget(close_btn)

        layout_grid  =  QtWidgets.QGridLayout()
        layout_grid.addWidget(ksf_h_lbl, 0, 0)
        layout_grid.addWidget(ksf_h_edt, 0, 1)
        layout_grid.addWidget(ksf_w_lbl, 1, 0)
        layout_grid.addWidget(ksf_w_edt, 1, 1)
        layout_grid.addWidget(nuclei_channel_lbl, 2, 0)
        layout_grid.addWidget(nuclei_channel_combo, 2, 1)
        layout_grid.addWidget(spots_channel_lbl, 3, 0)
        layout_grid.addWidget(spots_channel_combo, 3, 1)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addLayout(layout_grid)
        layout.addStretch()
        layout.addLayout(btns_box)

        self.setLayout(layout)
        self.setGeometry(300, 300, 60, 60)
        self.setWindowTitle("Settings Tool")

    def ksf_h_var(self, text):
        """Set keys size factor value (hight)."""
        self.ksf_h  =  np.float64(text)

    def ksf_w_var(self, text):
        """Set keys size factor value (width)."""
        self.ksf_w  =  np.float64(text)

    def nuclei_channel_switch(self, text):
        """Set nuclei channel."""
        self.nuclei_channel  =  int(text)

    def spots_channel_switch(self, text):
        """Set spots channel."""
        self.spots_channel  =  int(text)

    def save_vars(self):
        """Save new settings."""
        np.save('keys_size_factor.npy', [self.ksf_h, self.ksf_w])
        np.save('nucs_spts_default_channels.npy', [self.nuclei_channel, self.spots_channel])

    def close_(self):
        """Close the widget."""
        self.close()

    @QtCore.pyqtSlot()
    def restart(self):
        """Send message to main GUI."""
        self.procStart.emit()


class SetChannels(QtWidgets.QDialog):
    """Set the color channels of the raw data to put in the gui."""
    def __init__(self, parent=None):
        super().__init__(parent)

        ksf_h                       =  np.load('keys_size_factor.npy')[0]
        ksf_w                       =  np.load('keys_size_factor.npy')[1]
        spts_nucs_default_channels  =  np.load('nucs_spts_default_channels.npy')

        nuclei_channel_combo  =  QtWidgets.QComboBox(self)
        nuclei_channel_combo.addItem("None")
        nuclei_channel_combo.addItem("1")
        nuclei_channel_combo.addItem("2")
        nuclei_channel_combo.activated[str].connect(self.nuclei_channel_switch)
        nuclei_channel_combo.setCurrentIndex(spts_nucs_default_channels[0])
        nuclei_channel_combo.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))
        self.nuclei_channel_switch(nuclei_channel_combo.currentText())

        spots_channel_combo  =  QtWidgets.QComboBox(self)
        spots_channel_combo.addItem("None")
        spots_channel_combo.addItem("1")
        spots_channel_combo.addItem("2")
        spots_channel_combo.activated[str].connect(self.spots_channel_switch)
        spots_channel_combo.setCurrentIndex(spts_nucs_default_channels[1])
        spots_channel_combo.setFixedSize(int(ksf_h * 65), int(ksf_w * 25))
        self.spots_channel_switch(spots_channel_combo.currentText())

        nuclei_channel_lbl  =  QtWidgets.QLabel("Nuclei Channel", self)
        nuclei_channel_lbl.setFixedSize(int(ksf_h * 120), int(ksf_w * 22))

        spots_channel_lbl  =  QtWidgets.QLabel("Spots Channel", self)
        spots_channel_lbl.setFixedSize(int(ksf_h * 120), int(ksf_w * 22))

        nuclei_box  =  QtWidgets.QHBoxLayout()
        nuclei_box.addWidget(nuclei_channel_lbl)
        nuclei_box.addWidget(nuclei_channel_combo)

        spots_box  =  QtWidgets.QHBoxLayout()
        spots_box.addWidget(spots_channel_lbl)
        spots_box.addWidget(spots_channel_combo)

        enter_values_btn  =  QtWidgets.QPushButton("OK", self)
        enter_values_btn.setToolTip("Set Channels Number")
        enter_values_btn.setFixedSize(int(ksf_h * 60), int(ksf_w * 25))
        enter_values_btn.clicked.connect(self.enter_values)

        enter_box  =  QtWidgets.QHBoxLayout()
        enter_box.addStretch()
        enter_box.addWidget(enter_values_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addLayout(nuclei_box)
        layout.addLayout(spots_box)
        layout.addLayout(enter_box)

        self.chs_nucs_spts  =  np.copy(spts_nucs_default_channels)

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle("Set Channels")

    def nuclei_channel_switch(self, text):
        """Set nuclei channel."""
        self.nuclei_channel  =  int(text)

    def spots_channel_switch(self, text):
        """Set spots channel."""
        self.spots_channel  =  int(text)

    def enter_values(self):
        """Organizing channels info for the output."""
        if self.nuclei_channel != "None":
            self.chs_nucs_spts[0]  =  int(self.nuclei_channel)
        else:
            self.chs_nucs_spts[0]  =  0

        if self.spots_channel != "None":
            self.chs_nucs_spts[1]  =  int(self.spots_channel)
        else:
            self.chs_nucs_spts[1]  =  0

        self.chs_nucs_spts  =  np.asarray(self.chs_nucs_spts)

        self.close()

    def params(self):
        """Function to send results."""
        return self.chs_nucs_spts

    @staticmethod
    def getChannels(parent=None):
        """Send results."""
        dialog  =  SetChannels(parent)
        result  =  dialog.exec_()
        flag    =  dialog.params()
        return flag


def main():
    app         =  QtWidgets.QApplication(sys.argv)
    splash_pix  =  QtGui.QPixmap('Icons/DrosophilaIcon.png')
    splash      =  QtWidgets.QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    app.processEvents()
    ex  =  MainWindow()
    splash.finish(ex)
    sys.exit(app.exec_())


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':

    main()
    sys.excepthook  =  except_hook
