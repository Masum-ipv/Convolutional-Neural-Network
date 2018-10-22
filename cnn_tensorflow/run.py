import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import showImage


#Data Preparation
FolderNames = []
trainData = []
responseData = []
testData=[]
testResponseData=[]
for i in range(10):
    FolderNames.append(path+str(i))
#print(FolderNames)


k=0
for folder in FolderNames:
    k=k+1
    for image in listdir(folder):
        img =cv2.imread(join(folder,image))
        img = cv2.resize(img,(32,32))
        trainData.append(img.flatten())
        responseData.append(k)
#print(trainData)
#print(responseData)