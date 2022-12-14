# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HW1UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(787, 453)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(230, 40, 261, 341))
        self.groupBox.setObjectName("groupBox")
        self.splitter = QtWidgets.QSplitter(self.groupBox)
        self.splitter.setGeometry(QtCore.QRect(30, 50, 201, 271))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.cs = QtWidgets.QPushButton(self.splitter)
        self.cs.setObjectName("cs")
        self.ct = QtWidgets.QPushButton(self.splitter)
        self.ct.setObjectName("ct")
        self.cd = QtWidgets.QPushButton(self.splitter)
        self.cd.setObjectName("cd")
        self.bl = QtWidgets.QPushButton(self.splitter)
        self.bl.setObjectName("blending")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(510, 40, 241, 351))
        self.groupBox_2.setObjectName("groupBox_2")
        self.splitter_2 = QtWidgets.QSplitter(self.groupBox_2)
        self.splitter_2.setGeometry(QtCore.QRect(30, 60, 171, 231))
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.setObjectName("splitter_2")
        self.gb = QtWidgets.QPushButton(self.splitter_2)
        self.gb.setObjectName("gb")
        self.bf = QtWidgets.QPushButton(self.splitter_2)
        self.bf.setObjectName("bf")
        self.mf = QtWidgets.QPushButton(self.splitter_2)
        self.mf.setObjectName("mf")
        self.lm1 = QtWidgets.QPushButton(self.centralwidget)
        self.lm1.setGeometry(QtCore.QRect(30, 90, 150, 50))
        self.lm1.setObjectName("lm1")
        self.lm2 = QtWidgets.QPushButton(self.centralwidget)
        self.lm2.setGeometry(QtCore.QRect(30, 210, 150, 50))
        self.lm2.setObjectName("lm2")
        self.label_lm1 = QtWidgets.QLabel(self.centralwidget)
        self.label_lm1.setGeometry(QtCore.QRect(30, 140, 160, 60))
        self.label_lm1.setObjectName("label_lm1")
        self.label_lm1_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_lm1_2.setGeometry(QtCore.QRect(30, 260, 160, 60))
        self.label_lm1_2.setObjectName("label_lm1_2")
        mainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("mainWindow", "1. Image Processing"))
        self.cs.setText(_translate("mainWindow", "1.1 Color Separation "))
        self.ct.setText(_translate("mainWindow", "1.2 Color Transformation "))
        self.cd.setText(_translate("mainWindow", "1.3 Color Detection "))
        self.bl.setText(_translate("mainWindow", "1.4 Blending"))
        self.groupBox_2.setTitle(_translate("mainWindow", "2. Image Smoothing"))
        self.gb.setText(_translate("mainWindow", "2.1 Gaussian Blur"))
        self.bf.setText(_translate("mainWindow", "2.2 Bilateral Filter"))
        self.mf.setText(_translate("mainWindow", "2.3 Median Filter"))
        self.lm1.setText(_translate("mainWindow", "Load Image 1"))
        self.lm2.setText(_translate("mainWindow", "Load Image 2"))
        self.label_lm1.setText(_translate("mainWindow", "No image loaded"))
        self.label_lm1_2.setText(_translate("mainWindow", "No image loaded"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
