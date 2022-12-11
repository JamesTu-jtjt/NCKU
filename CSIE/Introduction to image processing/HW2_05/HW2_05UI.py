# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HW2_05UI.ui'
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
        self.groupBox.setGeometry(QtCore.QRect(20, 40, 231, 391))
        self.groupBox.setObjectName("groupBox")
        self.splitter = QtWidgets.QSplitter(self.groupBox)
        self.splitter.setGeometry(QtCore.QRect(0, 20, 221, 331))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.lm = QtWidgets.QPushButton(self.splitter)
        self.lm.setObjectName("lm")
        self.btn5_1 = QtWidgets.QPushButton(self.splitter)
        self.btn5_1.setObjectName("btn5_1")
        self.btn5_2 = QtWidgets.QPushButton(self.splitter)
        self.btn5_2.setObjectName("btn5_2")
        self.btn5_3 = QtWidgets.QPushButton(self.splitter)
        self.btn5_3.setObjectName("btn5_3")
        self.btn5_4 = QtWidgets.QPushButton(self.splitter)
        self.btn5_4.setObjectName("btn5_4")
        self.btn5_5 = QtWidgets.QPushButton(self.splitter)
        self.btn5_5.setObjectName("btn5_5")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(280, 50, 480, 380))
        self.label.setMaximumSize(QtCore.QSize(480, 380))
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_pred = QtWidgets.QLabel(self.centralwidget)
        self.label_pred.setGeometry(QtCore.QRect(650, 200, 200, 500))
        self.label_pred.setMaximumSize(QtCore.QSize(200, 200))
        self.label_pred.setText("fhfhfh")
        self.label_pred.setObjectName("label")
        mainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("mainWindow", "5. ResNet50"))
        self.lm.setText(_translate("mainWindow", "Load Image"))
        self.btn5_1.setText(_translate("mainWindow", "1. Show Image"))
        self.btn5_2.setText(_translate("mainWindow", "2. Show Distribution"))
        self.btn5_3.setText(_translate("mainWindow", "3. Show Model Structure"))
        self.btn5_4.setText(_translate("mainWindow", "4. Show Comparision"))
        self.btn5_5.setText(_translate("mainWindow", "5. Inference"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
