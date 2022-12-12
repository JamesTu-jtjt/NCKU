from email.mime import image
from heapq import merge
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import HW1_UI
import numpy as np
import cv2

class HW1_1_main(HW1_UI.Ui_mainWindow):    
    def __init__(self,MainWindow):
        super().setupUi(mainWindow)
        self.lm1.clicked.connect(self.load_image)
        self.lm2.clicked.connect(self.load_image_2)
        self.cs.clicked.connect(self.color_separation)
        self.ct.clicked.connect(self.color_transformation)
        self.cd.clicked.connect(self.color_detection)
        self.bl.clicked.connect(self.blending)
        self.gb.clicked.connect(self.gaussian_blur)
        self.bf.clicked.connect(self.bilateral_filter)
        self.mf.clicked.connect(self.median_filter)
        
    def load_image(self):
        img = cv2.imread('./Dataset_OpenCvDl_Hw1/Q1_Image/OpenCV.jpg')
        cv2.namedWindow('Load Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Load Image',img)
        self.label_lm1.setText("./Dataset_OpenCvDl_Hw1/Q1_Image/OpenCV.jpg")
        cv2.waitKey(0)
        cv2.destroyWindow('Load Image')
        
        if cv2.destroyWindow:
            self.label_lm1.setText("No Image Load")
            
    def load_image_2(self):
        img = cv2.imread('./Dataset_OpenCvDl_Hw1/Q1_Image/OpenCV.jpg')
        cv2.namedWindow('Load Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Load Image',img)
        self.label_lm1_2.setText("./Dataset_OpenCvDl_Hw1/Q1_Image/OpenCV.jpg")
        cv2.waitKey(0)
        cv2.destroyWindow('Load Image')
        
        if cv2.destroyWindow:
            self.label_lm1_2.setText("No Image Load")    
    
    def color_separation(self):
        img = cv2.imread('./Dataset_OpenCvDl_Hw1/Q1_Image/OpenCV.jpg')
        cv2.namedWindow('Color Separation', cv2.WINDOW_NORMAL)
        cv2.imshow('Color Separation',img)
        self.label_lm1.setText("./Dataset_OpenCvDl_Hw1/Q1_Image/OpenCV.jpg")
        b,g,r=cv2.split(img)
        zeros = np.zeros(img.shape[:2],dtype="uint8")
        cv2.namedWindow('cs_B', cv2.WINDOW_NORMAL)
        cv2.imshow('cs_B',cv2.merge([b,zeros,zeros]))
        cv2.namedWindow('cs_G', cv2.WINDOW_NORMAL)
        cv2.imshow('cs_G', cv2.merge([zeros,g,zeros]))
        cv2.namedWindow('cs_R', cv2.WINDOW_NORMAL)
        cv2.imshow('cs_R',cv2.merge([zeros,zeros,r])) 
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if cv2.destroyWindow:
            self.label_lm1.setText("No Image Load")
        
    def color_transformation(self):
        img = cv2.imread('./Dataset_OpenCvDl_Hw1/Q1_Image/OpenCV.jpg')
        cv2.namedWindow('Color Transformation', cv2.WINDOW_NORMAL)
        cv2.imshow('Color Transformation',img)
        self.label_lm1.setText("./Dataset_OpenCvDl_Hw1/Q1_Image/OpenCV.jpg")
        
        #using python function
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.namedWindow('Gray', cv2.WINDOW_NORMAL)
        cv2.imshow('Gray',img_gray)
        
        #separate and merge
        b,g,r = cv2.split(img)
        new = b/3+g/3+r/3
        new = np.uint8(new)
        cv2.namedWindow('Merged', cv2.WINDOW_NORMAL)
        cv2.imshow("Merged",new)
            
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if cv2.destroyWindow:
            self.label_lm1.setText("No Image Load")
        
    def color_detection(self):
        img = cv2.imread('./Dataset_OpenCvDl_Hw1/Q1_Image/OpenCV.jpg')
        cv2.namedWindow('Color Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Color Detection',img)
        self.label_lm1.setText("./Dataset_OpenCvDl_Hw1/Q1_Image/OpenCV.jpg")
        
        # Green detection
        img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        g_lowerbound = np.array([40,50,20])
        g_upperbound = np.array([80,255,255])
        g_mask =cv2.inRange(img_hsv,g_lowerbound,g_upperbound)
        g_masked = cv2.bitwise_and(img,img,mask=g_mask)
        cv2.namedWindow('Green mask', cv2.WINDOW_NORMAL)
        cv2.imshow('Green mask',g_mask)
        cv2.namedWindow('Green', cv2.WINDOW_NORMAL)
        cv2.imshow('Green',g_masked)
        
        # White detection
        w_lowerbound = np.array([0,0,200])
        w_upperbound = np.array([180,20,255])
        w_mask =cv2.inRange(img_hsv,w_lowerbound,w_upperbound)
        w_masked = cv2.bitwise_and(img,img,mask=w_mask)
        cv2.namedWindow('White mask', cv2.WINDOW_NORMAL)
        cv2.imshow('White mask',w_mask)
        cv2.namedWindow('White', cv2.WINDOW_NORMAL)
        cv2.imshow('White',w_masked)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if cv2.destroyWindow:
            self.label_lm1.setText("No Image Load")
    
    def blending(self): 
        img1 = cv2.imread('./Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Weak.jpg')
        cv2.namedWindow('Dog Weak', cv2.WINDOW_NORMAL)
        cv2.imshow('Dog Weak',img1) 
        img2 = cv2.imread('./Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Strong.jpg')
        cv2.namedWindow('Dog Strong', cv2.WINDOW_NORMAL)
        cv2.imshow('Dog Strong',img2)  
        self.label_lm1.setText("./Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Weak.jpg")
        self.label_lm1_2.setText("./Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Strong.jpg")
        
        def changeweight(val):
            alpha = val / 255
            beta = (1.0 - alpha)
            blending = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
            cv2.imshow('Blending', blending)
            
        cv2.namedWindow('Blending', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Blend bar', 'Blending', 0, 255, changeweight)
        cv2.setTrackbarPos('Blend bar','Blending', 127)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if cv2.destroyWindow:
            self.label_lm1.setText("No Image Load")
            self.label_lm1_2.setText("No Image Load")
    
    def gaussian_blur(self):
        img = cv2.imread('./Dataset_OpenCvDl_Hw1/Q2_Image/image1.jpg')
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.imshow('Original',img) 
        self.label_lm1.setText("./Dataset_OpenCvDl_Hw1/Q2_Image/image1.jpg")
        
        def changmagnitude(val):
            k= val*2+1
            blur = cv2.GaussianBlur(img,(k,k),0) 
            cv2.imshow('Gaussian Blur', blur)
            
        cv2.namedWindow('Gaussian Blur', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('magnitude', 'Gaussian Blur', 0, 10, changmagnitude)
        cv2.setTrackbarPos('magnitude','Gaussian Blur', 5)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if cv2.destroyWindow:
            self.label_lm1.setText("No Image Load")
        
    def bilateral_filter(self):
        img = cv2.imread('./Dataset_OpenCvDl_Hw1/Q2_Image/image1.jpg')
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.imshow('Original',img)     
        self.label_lm1.setText("./Dataset_OpenCvDl_Hw1/Q2_Image/image1.jpg")
        
        def changmagnitude(val):
            k= val*2+1  
            filter = cv2.bilateralFilter(img, k, 90, 90) 
            cv2.imshow('Bilateral Filter', filter)
        
        cv2.namedWindow('Bilateral Filter', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('magnitude', 'Bilateral Filter', 0, 10, changmagnitude)
        cv2.setTrackbarPos('magnitude','Bilateral Filter', 5)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if cv2.destroyWindow:
            self.label_lm1.setText("No Image Load")
        
    def median_filter(self):
        img = cv2.imread('./Dataset_OpenCvDl_Hw1/Q2_Image/image2.jpg')
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.imshow('Original',img)     
        self.label_lm1.setText("./Dataset_OpenCvDl_Hw1/Q2_Image/image2.jpg")
        
        def changmagnitude(val):
            k= val*2+1 
            filter = cv2.medianBlur(img,k) 
            cv2.imshow('Median Filter', filter)
        
        cv2.namedWindow('Median Filter', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('magnitude', 'Median Filter', 0, 10, changmagnitude)
        cv2.setTrackbarPos('magnitude','Median Filter', 5)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if cv2.destroyWindow:
            self.label_lm1.setText("No Image Load")
        
              
if __name__ == '__main__':  
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = HW1_1_main(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())