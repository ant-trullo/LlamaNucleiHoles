"""This function contains several usefull widgets.
"""


import numpy as np
from PyQt5 import QtWidgets  # , Qt
from PyQt5.QtCore import Qt


class SetPixelSize(QtWidgets.QDialog):
    """Set manually the pixel size."""
    def __init__(self, parent=None):
        super().__init__(parent)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        size_xy_lbl  =  QtWidgets.QLabel("X-Y size (µ)", self)
        size_xy_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 22))

        size_xy_edt  =  QtWidgets.QLineEdit()
        size_xy_edt.setFixedSize(int(ksf_h * 50), int(ksf_w * 22))
        size_xy_edt.textChanged[str].connect(self.size_xy_var)

        size_xy_box  =  QtWidgets.QHBoxLayout()
        size_xy_box.addWidget(size_xy_lbl)
        size_xy_box.addWidget(size_xy_edt)

        size_z_lbl  =  QtWidgets.QLabel("Z size (µ)", self)
        size_z_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 22))

        size_z_edt  =  QtWidgets.QLineEdit()
        size_z_edt.setFixedSize(int(ksf_h * 50), int(ksf_w * 22))
        size_z_edt.textChanged[str].connect(self.size_z_var)

        size_z_box  =  QtWidgets.QHBoxLayout()
        size_z_box.addWidget(size_z_lbl)
        size_z_box.addWidget(size_z_edt)

        send_btn  =  QtWidgets.QPushButton("Ok")
        send_btn.clicked.connect(self.send)
        send_btn.setToolTip("Insert values")
        send_btn.setFixedSize(int(ksf_h * 60), int(ksf_w * 25))

        send_box  =  QtWidgets.QHBoxLayout()
        send_box.addStretch()
        send_box.addWidget(send_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addLayout(size_xy_box)
        layout.addLayout(size_z_box)
        layout.addLayout(send_box)

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, int(ksf_h * 20), int(ksf_w * 25))
        self.setWindowTitle("Set Pixel Size")

    def size_xy_var(self, text):
        """Set X-Y pixel size."""
        self.size_xy_value  =  float(text)

    def size_z_var(self, text):
        """Set Z pixel size."""
        self.size_z_value  =  float(text)

    def params(self):
        """Function to send choice."""
        return [self.size_xy_value, self.size_z_value]

    def send(self):
        """Input values."""
        self.close()

    @staticmethod
    def getPixelsValues(parent=None):
        """Send choice."""
        dialog  =  SetPixelSize(parent)
        result  =  dialog.exec_()
        sizess  =  dialog.params()
        return sizess


class ProgressBar(QtWidgets.QWidget):
    """Simple progress bar widget"""
    def __init__(self, parent=None, total1=20):
        super().__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar  =  QtWidgets.QProgressBar()
        self.progressbar.setMinimum(1)
        self.progressbar.setMaximum(total1)

        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)

    def update_progressbar(self, val1):
        """update the progressbar"""
        self.progressbar.setValue(val1)
        QtWidgets.qApp.processEvents()


class ProgressBarDouble(QtWidgets.QWidget):
    """Double Progressbar widget."""
    def __init__(self, parent=None, total1=20, total2=20):
        super().__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar1  =  QtWidgets.QProgressBar()
        self.progressbar1.setMinimum(1)
        self.progressbar1.setMaximum(total1)

        self.progressbar2  =  QtWidgets.QProgressBar()
        self.progressbar2.setMinimum(1)
        self.progressbar2.setMaximum(total2)

        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar1, 0, 0)
        main_layout.addWidget(self.progressbar2, 1, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)


    def update_progressbar1(self, val1):
        """Update progressbar 1."""
        self.progressbar1.setValue(val1)
        QtWidgets.qApp.processEvents()


    def update_progressbar2(self, val2):
        """Update progressbar 2."""
        self.progressbar2.setValue(val2)
        QtWidgets.qApp.processEvents()


    def pbar2_setmax(self, total2):
        self.progressbar2.setMaximum(total2)


class ProgressBarTriple(QtWidgets.QWidget):
    """Double Progressbar widget."""
    def __init__(self, parent=None, total1=20, total2=20, total3=20):
        super().__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar1  =  QtWidgets.QProgressBar()
        self.progressbar1.setMinimum(1)
        self.progressbar1.setMaximum(total1)

        self.progressbar2  =  QtWidgets.QProgressBar()
        self.progressbar2.setMinimum(1)
        self.progressbar2.setMaximum(total2)

        self.progressbar3  =  QtWidgets.QProgressBar()
        self.progressbar3.setMinimum(1)
        self.progressbar3.setMaximum(total3)

        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar1, 0, 0)
        main_layout.addWidget(self.progressbar2, 1, 0)
        main_layout.addWidget(self.progressbar3, 2, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)

    def update_progressbar1(self, val1):
        """Update progressbar 1."""
        self.progressbar1.setValue(val1)
        QtWidgets.qApp.processEvents()

    def update_progressbar2(self, val2):
        """Update progressbar 2."""
        self.progressbar2.setValue(val2)
        QtWidgets.qApp.processEvents()

    def update_progressbar3(self, val3):
        """Update progressbar 2."""
        self.progressbar3.setValue(val3)
        QtWidgets.qApp.processEvents()

    # def pbar2_setmax(self, total2):
    #     self.progressbar2.setMaximum(total2)
