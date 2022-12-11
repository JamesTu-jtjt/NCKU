from email.mime import image
from fileinput import filename
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap,QImage
import cv2 
import sys
import HW2_05UI
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
import random
import cv2
from PIL import Image
import os
import func


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.Resize([224,224]),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder('./Dataset_OpenCvDl_Hw2_Q5/training_dataset',transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder('./Dataset_OpenCvDl_Hw2_Q5/validation_dataset', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)

classes = ('Cat', 'Dog')

model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(nn.Linear(2048,1),nn.Sigmoid())
model.to(device)

PATH = './resnet_model_BCE.pth'
model.load_state_dict(torch.load(PATH)) 


def show_image():
    infernceset = torchvision.datasets.ImageFolder('./Dataset_OpenCvDl_Hw2_Q5/inference_dataset',transform=transform)
    a = random.randint(1,4)
    b = random.randint(5,8)
    plt.figure(num='Figur 1',figsize=(10,10))
    plt.subplot(1,2,1)
    plt.title('cat')
    plt.imshow(infernceset[a][0].permute(1,2,0))
    plt.subplot(1,2,2)
    plt.title('Dog')
    plt.imshow(infernceset[b][0].permute(1,2,0))
    plt.show()

def show_distribution():
    dis = cv2.imread("./cd.png") 
    cv2.imshow("Figure 1",dis)   
    
def show_structure():
    summary(model, (3,224,224))

def show_comparision():
    dis = cv2.imread("./ac.png") 
    cv2.imshow("Figure 1",dis) 



class HW2_05_main(HW2_05UI.Ui_mainWindow):
    
    def __init__(self,MainWindow):
        super().setupUi(mainWindow)
        self.lm.clicked.connect(self.load_image)
        self.btn5_1.clicked.connect(show_image)
        self.btn5_2.clicked.connect(show_distribution)
        self.btn5_3.clicked.connect(show_structure)
        self.btn5_4.clicked.connect(show_comparision)
        self.btn5_5.clicked.connect(self.inference)
        
    def load_image(self,filepath):
        filename, _ = QFileDialog.getOpenFileName(None,
                  "open file",
                  "","Image Files (*.jpg *.png)") 
        self.image_PIL = Image.open(filename)
        scaled = QPixmap(filename).scaled(350,350)
        self.label.setPixmap(scaled)
        
    def inference(self):    
        img = self.image_PIL
        
        img = img.resize((224, 224))
        image_tensor = transform(img)
        image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.to(device)
        out = model(image_tensor)
        thresh = 0.5
        
        if(out > thresh):
            self.label_pred.setText('Prediction : Dog')
        else:
            self.label_pred.setText('Prediction : Cat')
        
         
    
if __name__ == '__main__':  
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = HW2_05_main(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())