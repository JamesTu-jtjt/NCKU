from email.mime import image
from fileinput import filename
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap,QImage
import cv2 
import sys
import HW1_05UI
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import random 
from torchsummary import summary
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                        shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                        shuffle=False,num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model = models.vgg19(pretrained=False)
model.to(device)

PATH = './cifar_model.pth'
model.load_state_dict(torch.load(PATH)) 


def show_train_data():
    random_data = random.sample(range(50000),9)
    k = 1
    plt.figure(num='Figur 1',figsize=(10,10))
    for i in random_data:
        plt.subplot(3,3,k)     
        plt.title(classes[trainset.targets[i]])   
        plt.imshow(trainset.data[i,:])      
        plt.axis('off')     
        k += 1
    plt.show()
    
def show_structure():
    summary(model, (3,32,32))

def show_data_augmentation():
    oringin = trainset.data[random.randint(0, 9)]
    img = oringin
    transform_list = ([transforms.CenterCrop(20),transforms.Grayscale(num_output_channels=3),transforms.RandomAffine(degrees= 20 ,shear=(70))])
    transform_title = (['CenterCrop','Grayscale','RandomAffine'])
    k = 1
    plt.figure(num='Figur 1',figsize=(10,10))
    for i in transform_list:
        plt.subplot(1,3,k) 
        PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')
        transform0 = transforms.Compose([i])
        img = transform0(PIL_image)
        plt.title(transform_title[k-1] )
        plt.imshow(img)      
        plt.axis('off')     
        k += 1
        img = oringin
    plt.show()

def show_accuracy_Loss():
    acc = cv2.imread('./acc.png')
    loss = cv2.imread('./Loss.png')
    cv2.imshow("Accuracy",acc)
    cv2.imshow("Loss",loss)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

class HW1_05_main(HW1_05UI.Ui_mainWindow):
    
    def __init__(self,MainWindow):
        super().setupUi(mainWindow)
        self.lm.clicked.connect(self.load_image)
        self.btn5_1.clicked.connect(show_train_data)
        self.btn5_2.clicked.connect(show_structure)
        self.btn5_3.clicked.connect(show_data_augmentation)
        self.btn5_4.clicked.connect(show_accuracy_Loss)
        self.btn5_5.clicked.connect(self.inference)
        
    def load_image(self,filepath):
        filename, _ = QFileDialog.getOpenFileName(None,
                  "open file",
                  "","Image Files (*.jpg *.png)") 
        self.image_PIL = Image.open(filename)
        scaled = QPixmap(filename).scaled(400,400)
        self.label.setPixmap(scaled)
        
    def inference(self):    
        img = self.image_PIL
        
        img = img.resize((32, 32))
        image_tensor = transform(img)
        image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.to(device)
        out = model(image_tensor)
        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0]
        labelname = classes[(indices[0][0])]
        percent = percentage[(indices[0][0])].item()
        percent = format(percent, '.2f')

        plt.title('confidence: {percentage} \n Prediction Label: {label}'.format(percentage=percent, label=labelname))
        plt.imshow(img)
        plt.show()
        
         
    
if __name__ == '__main__':  
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = HW1_05_main(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())