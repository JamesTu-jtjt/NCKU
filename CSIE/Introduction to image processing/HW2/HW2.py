from ctypes.wintypes import HENHMETAFILE
from math import sqrt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import sys
import HW2UI
import numpy as np
import cv2
import glob
import time
import python_utils as utils
class HW1_2_main(HW2UI.Ui_mainWindow):    
    def __init__(self,MainWindow):
        super().setupUi(mainWindow)
        self.btn_loadfolder.clicked.connect(self.load_folder)
        self.btn_load1.clicked.connect(self.load_image_L)
        self.btn_load2.clicked.connect(self.load_image_R)
        self.btn_1_1.clicked.connect(self.draw_contour)
        self.btn_1_2.clicked.connect(self.count_rings)
        self.btn_2_1.clicked.connect(self.corner_detection)
        self.btn_2_2.clicked.connect(self.find_instrinsic)
        self.btn_2_3.clicked.connect(self.find_extrinsic)
        self.btn_2_4.clicked.connect(self.find_distortion)
        self.btn_2_5.clicked.connect(self.show_undistorted)
        self.btn_3_1.clicked.connect(self.word_onboard)
        self.btn_3_2.clicked.connect(self.word_onvertical)
        self.btn_4_1.clicked.connect(self.stereo_disparity)
    
    def load_folder(self,filepath): 
        self.foldername = QFileDialog.getExistingDirectory(None,"select folder","")
        self.label_folder.setText(self.foldername)
    
    def load_image_L(self,filepath):
        filename, _ = QFileDialog.getOpenFileName(None,
                  "open file",
                  "","Image Files (*.jpg *.png)") 
        self.label_load1.setText(filename)
        self.img1 = cv2.imread(filename)
        # cv2.imshow('Load Image L', self.img1) 
    
    def load_image_R(self,filepath):
        filename, _ = QFileDialog.getOpenFileName(None,
                  "open file",
                  "","Image Files (*.jpg *.png)") 
        self.label_load2.setText(filename)
        self.img2 = cv2.imread(filename)
        # cv2.namedWindow('Load Image R')
        # cv2.resizeWindow('Load Image R',800,600)
        # cv2.imshow('Load Image R', self.img2)
        
    # Q1    
    def draw_contour(self):
        # img1
        gray_image = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        (thresh, binary) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        guassian = cv2.GaussianBlur(binary, (11, 11), 0)
        edge_image = cv2.Canny(guassian, 127, 127)

        edge_image, self.contours1, hierarchy = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_show1 = self.img1.copy()
        
        cv2.drawContours(image_show1, self.contours1, -1, (0, 0, 255), 2)
        cv2.namedWindow('image_copy1')
        cv2.imshow('image_copy1', image_show1)
        
        # img2
        gray_image = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        (thresh, binary) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        guassian = cv2.GaussianBlur(binary, (11, 11), 0)
        edge_image = cv2.Canny(guassian, 127, 127)

        edge_image, self.contours2, hierarchy = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_show2 = self.img2.copy()
        
        cv2.drawContours(image_show2, self.contours2, -1, (0, 0, 255), 2)
        cv2.namedWindow('image_copy2')
        cv2.imshow('image_copy2', image_show2)
        
    def count_rings(self):
        count1 = len(self.contours1) 
        self.label_count1.setText("There are {} rings in img1.jpg".format(count1))
        count2 = len(self.contours2)
        self.label_count2.setText("There are {} rings in img2.jpg".format(count2))
    
    # Q2    
    def corner_detection(self):
        
        imgs = glob.glob(self.foldername + '/*.bmp')
        for i in range(len(imgs)):
            img = cv2.imread(imgs[i])
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ny = 8
            nx = 11
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            # If found, draw corners
            if ret == True:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                cv2.namedWindow("{}".format((i+1)))
                img_re= cv2.resize(img, (960, 540))  
                cv2.imshow("{}".format((i+1)), img_re)
                key = cv2.waitKey(500)
                cv2.destroyWindow("{}".format((i+1)))
             
    def find_instrinsic(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)
        
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        
        imgs = glob.glob(self.foldername + '/*.bmp')
        
        for i in range(len(imgs)):
            img = cv2.imread(imgs[i])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
                
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
        self.cameraMatrix=mtx
        self.disMatrix=dist
        self.rot=rvecs
        self.tran=tvecs
        print("Intrinsic :")   
        print(mtx)
        
    def find_extrinsic(self):
        value = int(self.comboBox.currentText()) -1
        rvec = self.rot[value]
        tvec = self.tran[value]
        rotMatrix,_ = cv2.Rodrigues(rvec)

        output=[]
        for i in range(len(rotMatrix)):
            for j in rotMatrix[i]:
                output.append(j)
            output.append(tvec[i, 0])
        print("Extrinsic :")    
        print("[[", end="")
        for i in range(len(output)):
            if i == 3 or i == 7:
                print(output[i], end="]\n")
            elif i == 11:
                print(output[i], end="]")
            elif i==4 or i== 8:
                print(" [{}".format(output[i]),end="\t")
            else:
                print(output[i], end="\t")
        print("]")
                       
    def find_distortion(self):
        print("Distortion :")
        for i in self.disMatrix:
            print(i)  
    
    def show_undistorted(self):
        
        imgs = glob.glob(self.foldername + '/*.bmp')
        for i in range(len(imgs)):
            img = cv2.imread(imgs[i])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            undistorted = cv2.undistort(img,self.cameraMatrix,self.disMatrix,None,self.cameraMatrix)
            cv2.namedWindow("{} distorted".format((i+1)))
            cv2.namedWindow("{} undistorted".format((i+1)))
            cv2.moveWindow("{} distorted".format((i+1)), 450, 450)
            cv2.moveWindow("{} undistorted".format((i+1)),1450,450)
            img_re1= cv2.resize(gray, (960, 540))
            img_re2= cv2.resize(undistorted, (960, 540))
            cv2.imshow("{} distorted".format((i+1)), img_re1)  
            cv2.imshow("{} undistorted".format((i+1)), img_re2)
            key = cv2.waitKey(500)
            cv2.destroyAllWindows()
              
    # Q3
    def word_onboard(self):
        fs = cv2.FileStorage("./Dataset_OpenCvDl_Hw2/Q3_Image/Q2_Lib/alphabet_lib_onboard.txt", cv2.FILE_STORAGE_READ)
        string = self.textEdit.toPlainText().upper()
        imgsResult = []
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        axis = np.float32([[1, 1, 0], [5, 1, 0], [3, 5, 0], [3, 3, -3]]).reshape(-1, 3)
        imgs = glob.glob(self.foldername + '/*.bmp') 
        
        for i in range(len(imgs)):
            img = cv2.imread(imgs[i])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11,8), None) 
            
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)  
                
                
            cnt = 0
            for c in string:
                if(c.isalpha()):
                    ch = fs.getNode(c).mat()
                    posX = (cnt % 4) * 3
                    posY = int(cnt / 4) * 3

                    for y in ch:
                        src = np.array(y, np.float)
                        src[0][0] += 9 - posX
                        src[0][1] += 6 - posY
                        src[1][0] += 9 - posX
                        src[1][1] += 6 - posY
                        result = cv2.projectPoints(src, rvecs[i], tvecs[i], mtx, None)
                        result = tuple(map(tuple, result[0]))
                        start = tuple(map(int, result[0][0]))
                        end = tuple(map(int, result[1][0]))
                        cv2.line(img, start, end, (255, 0, 0), 10)
                    cnt += 1

            img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgsResult.append(img)
                
                
        for img in imgsResult:
            cv2.namedWindow("{}".format((i+1)))
            cv2.resizeWindow("{}".format((i+1)), 800, 600)
            cv2.imshow("{}".format((i+1)), img)
            key = cv2.waitKey(500)
            cv2.destroyWindow("{}".format((i+1)))
    
    def word_onvertical(self):
        fs = cv2.FileStorage("./Dataset_OpenCvDl_Hw2/Q3_Image/Q2_Lib/alphabet_lib_vertical.txt", cv2.FILE_STORAGE_READ)
        string = self.textEdit.toPlainText().upper()
        imgsResult = []
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        axis = np.float32([[1, 1, 0], [5, 1, 0], [3, 5, 0], [3, 3, -3]]).reshape(-1, 3)
        imgs = glob.glob(self.foldername + '/*.bmp') 
        
        for i in range(len(imgs)):
            img = cv2.imread(imgs[i])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11,8), None) 
            
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)  
                
                
            cnt = 0
            for c in string:
                if(c.isalpha()):
                    ch = fs.getNode(c).mat()
                    posX = (cnt % 4) * 3
                    posY = int(cnt / 4) * 3

                    for y in ch:
                        src = np.array(y, np.float)
                        src[0][0] += 9 - posX
                        src[0][1] += 6 - posY
                        src[1][0] += 9 - posX
                        src[1][1] += 6 - posY
                        result = cv2.projectPoints(src, rvecs[i], tvecs[i], mtx, None)
                        result = tuple(map(tuple, result[0]))
                        start = tuple(map(int, result[0][0]))
                        end = tuple(map(int, result[1][0]))
                        cv2.line(img, start, end, (255, 0, 0), 10)
                    cnt += 1

            img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgsResult.append(img)
                
                
        for img in imgsResult:
            cv2.namedWindow("{}".format((i+1)))
            cv2.resizeWindow("{}".format((i+1)), 800, 600)
            cv2.imshow("{}".format((i+1)), img)
            key = cv2.waitKey(500)
            cv2.destroyWindow("{}".format((i+1)))
            
    def stereo_disparity(self):
        imgL = cv2.cvtColor(self.img1,cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(self.img2,cv2.COLOR_BGR2GRAY)
        
        stereo = cv2.StereoBM_create(256,25)
        disparity = stereo.compute(imgL,imgR)
        disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disparity = cv2.resize(disparity, (800, 600), interpolation=cv2.INTER_AREA)
        cv2.namedWindow("Disparity")
        cv2.resizeWindow("Disparity", 800, 600)
        cv2.imshow("Disparity",disparity)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        cv2.waitKey(500)
        
        imgL_new = cv2.resize(self.img1, (800, 600), interpolation=cv2.INTER_AREA)
        imgR_new = cv2.resize(self.img2, (800, 600), interpolation=cv2.INTER_AREA)
        
        def OnMouseAction(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                dist = int(disparity[y][x] / 4)
                img = imgR_new.copy()
                img = cv2.circle(img, (x - dist, y), 5, (0, 0, 255), 10)
                cv2.imshow('imgR_dot', img)

        cv2.namedWindow('imgL')
        cv2.resizeWindow('imgL', 800, 600)
        cv2.setMouseCallback('imgL',OnMouseAction)
        cv2.imshow('imgL', imgL_new)
        cv2.namedWindow('imgR_dot')
        cv2.resizeWindow('imgR_dot', 800, 600)
        cv2.imshow('imgR_dot', imgR_new)
        
        
        
                
if __name__ == '__main__':  
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = HW1_2_main(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())