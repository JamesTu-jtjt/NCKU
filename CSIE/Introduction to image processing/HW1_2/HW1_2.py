from ctypes.wintypes import HENHMETAFILE
from math import sqrt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import sys
import HW1_2UI
import numpy as np
import cv2

class HW1_2_main(HW1_2UI.Ui_mainWindow):    
    def __init__(self,MainWindow):
        super().setupUi(mainWindow)
        self.lm.clicked.connect(self.load_image)
        self.btn3_1.clicked.connect(self.gaussian_blur)
        self.btn3_2.clicked.connect(self.sobel_x)
        self.btn3_3.clicked.connect(self.sobel_y)
        self.btn3_4.clicked.connect(self.magnitude)
        self.btn4_1.clicked.connect(self.resize)
        self.btn4_2.clicked.connect(self.translation)
        self.btn4_3.clicked.connect(self.rotation_scaling)
        self.btn4_4.clicked.connect(self.shearing)
    
       
    def load_image(self):
        
        filename, _ = QFileDialog.getOpenFileName(None,
                  "open file",
                  "","Image Files (*.jpg *.png)")               
        self.label_lm.setText(filename)
        
        img = cv2.imread(filename)
        cv2.imshow('Load Image', img) 
        
        
    def gaussian_blur(self):
        img = cv2.imread(self.label_lm.text())
        gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        width,height = gray.shape[:2]
        gb = np.zeros([width,height])
        normalize = np.array([[0.045,0.122,0.045],[0.122,0.332,0.122],[0.045,0.122,0.045]])
        
        #convolution
        for i in range(width-2):
            for j in range(height-2):
                gb[i,j] =  normalize[0,0]*gray[i,j]+normalize[0,1]*gray[i,j+1]+normalize[0,2]*gray[i,j+2]+normalize[1,0]*gray[i+1,j]+normalize[1,1]*gray[i+1,j+1]+normalize[1,2]*gray[i+1,j+2]+normalize[2,0]*gray[i+2,j]+normalize[2,1]*gray[i+2,j+1]+normalize[2,2]*gray[i+2,j+2]
        # print(gb.dtype)
        gb = gb.astype(np.uint8)
        cv2.imwrite('gaussian_blur.jpg', gb)   
        cv2.imshow('Gray', gray)         
        cv2.imshow('Gaussian Blur', gb)  
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        
        
    def sobel_x(self):
        
        img = cv2.imread(self.label_lm.text())
        gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        width,height = gray.shape[:2]
        gb = np.zeros([width,height])
        normalize = np.array([[0.045,0.122,0.045],[0.122,0.332,0.122],[0.045,0.122,0.045]])
        
        #convolution
        for i in range(width-2):
            for j in range(height-2):
                gb[i,j] =  normalize[0,0]*gray[i,j]+normalize[0,1]*gray[i,j+1]+normalize[0,2]*gray[i,j+2]+normalize[1,0]*gray[i+1,j]+normalize[1,1]*gray[i+1,j+1]+normalize[1,2]*gray[i+1,j+2]+normalize[2,0]*gray[i+2,j]+normalize[2,1]*gray[i+2,j+1]+normalize[2,2]*gray[i+2,j+2]
        # print(gb.dtype)
        gb = gb.astype(np.uint8)
        
        Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        sobelx = np.zeros([width-4,height-4])
        for i in range(width-4):
            for j in range(height-4):
                sobelx[i,j] = abs(gb[i,j]*Gx[0,0] + gb[i+1,j]*Gx[1,0] + gb[i+2,j]*Gx[2,0]+ gb[i,j+2]*Gx[0,2] + gb[i+1,j+2]*Gx[1,2] + gb[i+2,j+2]*Gx[2,2])
        
        Max = np.amax(sobelx)
        Min = np.amin(sobelx)
        
        for i in range(width-4):
            for j in range(457):
                sobelx[i,j] = 255*(sobelx[i,j]-Min)/(Max - Min)
                
        sobelx = sobelx.astype(np.uint8)
        cv2.imshow('Gaussian Blur', gb)   
        cv2.imshow('Sobel X', sobelx)  
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        
    def sobel_y(self):
        img = cv2.imread(self.label_lm.text())
        gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        width,height = gray.shape[:2]
        gb = np.zeros([width,height])
        normalize = np.array([[0.045,0.122,0.045],[0.122,0.332,0.122],[0.045,0.122,0.045]])
        
        #convolution
        for i in range(width-2):
            for j in range(height-2):
                gb[i,j] =  normalize[0,0]*gray[i,j]+normalize[0,1]*gray[i,j+1]+normalize[0,2]*gray[i,j+2]+normalize[1,0]*gray[i+1,j]+normalize[1,1]*gray[i+1,j+1]+normalize[1,2]*gray[i+1,j+2]+normalize[2,0]*gray[i+2,j]+normalize[2,1]*gray[i+2,j+1]+normalize[2,2]*gray[i+2,j+2]
        # print(gb.dtype)
        gb = gb.astype(np.uint8)
        
        Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        sobely = np.zeros([width-4,height-4])
        for i in range(width-4):
            for j in range(height-4):
                    sobely[i,j] = abs(gb[i,j]*Gy[0,0] + gb[i+1,j]*Gy[1,0] + gb[i+2,j]*Gy[2,0]+ gb[i,j+2]*Gy[0,2] + gb[i+1,j+2]*Gy[1,2] + gb[i+2,j+2]*Gy[2,2])

        Max = np.amax(sobely)
        Min = np.amin(sobely)
        
        for i in range(width-4):
            for j in range(height-4):
                sobely[i,j] = 255*(sobely[i,j]-Min)/(Max - Min)
                
        sobely = sobely.astype(np.uint8)
        cv2.imshow('Gaussian Blur', gb)   
        cv2.imshow('Sobel Y', sobely)  
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        
    def magnitude(self):
        img = cv2.imread(self.label_lm.text())
        gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        width,height = gray.shape[:2]
        gb = np.zeros([width,height])
        normalize = np.array([[0.045,0.122,0.045],[0.122,0.332,0.122],[0.045,0.122,0.045]])
        
        #convolution
        for i in range(width-2):
            for j in range(height-2):
                gb[i,j] =  normalize[0,0]*gray[i,j]+normalize[0,1]*gray[i,j+1]+normalize[0,2]*gray[i,j+2]+normalize[1,0]*gray[i+1,j]+normalize[1,1]*gray[i+1,j+1]+normalize[1,2]*gray[i+1,j+2]+normalize[2,0]*gray[i+2,j]+normalize[2,1]*gray[i+2,j+1]+normalize[2,2]*gray[i+2,j+2]
        # print(gb.dtype)
        gb = gb.astype(np.uint8)
        
        #sobel X
        Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        sobelx = np.zeros([width-4,height-4])
        for i in range(width-4):
            for j in range(height-4):
                sobelx[i,j] = abs(gb[i,j]*Gx[0,0] + gb[i+1,j]*Gx[1,0] + gb[i+2,j]*Gx[2,0]+ gb[i,j+2]*Gx[0,2] + gb[i+1,j+2]*Gx[1,2] + gb[i+2,j+2]*Gx[2,2])
        
        Max = np.amax(sobelx)
        Min = np.amin(sobelx)
        
        for i in range(246):
            for j in range(457):
                sobelx[i,j] = 255*(sobelx[i,j]-Min)/(Max - Min)
                
        sobelx = sobelx.astype(np.uint8)
        
        #sobel Y
        Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        sobely = np.zeros([width-4,height-4])
        for i in range(width-4):
            for j in range(height-4):
                    sobely[i,j] = abs(gb[i,j]*Gy[0,0] + gb[i+1,j]*Gy[1,0] + gb[i+2,j]*Gy[2,0]+ gb[i,j+2]*Gy[0,2] + gb[i+1,j+2]*Gy[1,2] + gb[i+2,j+2]*Gy[2,2])

        Max = np.amax(sobely)
        Min = np.amin(sobely)
        
        for i in range(width-4):
            for j in range(height-4):
                sobely[i,j] = 255*(sobely[i,j]-Min)/(Max - Min)
                
        sobely = sobely.astype(np.uint8)
        
        mag = np.zeros([width-4,height-4])
        for i in range(width-4):
            for j in range(height-4):
                mag[i,j] = sqrt(abs(sobelx[i,j]**2+sobely[i,j]**2))
        mag = mag.astype(np.uint8)
              
        cv2.imshow('Magnitude', mag)  
        cv2.waitKey(0)
        cv2.destroyAllWindows()
         
    def resize(self):
        img = cv2.imread(self.label_lm.text())
        re = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
        N = cv2.getRotationMatrix2D((0,0),0,1) #scaling
        re = cv2.warpAffine(re,N,(img.shape[0], img.shape[1]))
          
        cv2.namedWindow('Resize',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Resize',430,430)
        cv2.imwrite('Resize.jpg', re)
        cv2.imshow('Resize',re)
         
        
    def translation(self):
        img = cv2.imread(self.label_lm.text())
        re = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
        N = cv2.getRotationMatrix2D((0,0),0,1) 
        re = cv2.warpAffine(re,N,(img.shape[0], img.shape[1]))
        M = np.float32([[1,0,215], [0,1,215]])
        re1 =  cv2.warpAffine(re,M,(img.shape[0], img.shape[1]))
        merge = cv2.addWeighted(re,1,re1,1,0.0)
        cv2.namedWindow('Translation',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Translation',430,430)
        cv2.imwrite('Translation.jpg', merge)
        cv2.imshow('Translation',merge)
        
    def rotation_scaling(self):
        img = cv2.imread(self.label_lm.text())
        N = cv2.getRotationMatrix2D((215,215),45,0.5)
        re = cv2.warpAffine(img,N,(img.shape[0], img.shape[1]))
        cv2.namedWindow('Rotation Scaling',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Rotation Scaling',430,430)
        cv2.imwrite('Rotation Scaling.jpg', re)
        cv2.imshow('Rotation Scaling',re)   
        
    def shearing(self):
        img = cv2.imread(self.label_lm.text())
        width,height = img.shape[:2]
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[100,50],[100,250]])
        matrix = cv2.getAffineTransform(pts1,pts2)
        re = cv2.warpAffine(img, matrix, (width,height))
        cv2.namedWindow('Shearing',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Shearing',430,430)
        cv2.imwrite('Shearing.jpg', re)
        cv2.imshow('Shearing',re)     
        
if __name__ == '__main__':  
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = HW1_2_main(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())