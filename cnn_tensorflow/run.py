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
    FolderNames.append(path+str(i)) # Adding all the folder name to a list
#print(FolderNames)


k=0
for folder in FolderNames:
    for image in listdir(folder):
        img =cv2.imread(join(folder,image))
        img = cv2.resize(img,(32,32))
        trainData.append(img.flatten())
        responseData.append(k)
    k=k+1
trainNp = np.float32(trainData) # (2183, 3072)
responseNp = np.float32(responseData) # (2183,)


# OneHot Encoder
labels = tf.constant(responseData)
highest_label = tf.reduce_max(labels)  # highest_label 9
labels_one_hot = tf.one_hot(labels, highest_label + 1) # (2183, 10) 

all_data = tf.concat((trainNp, labels_one_hot), axis=1) # TensorShape([Dimension(2183), Dimension(3082)])
tf.random_shuffle(all_data, seed= 0) # TensorShape([Dimension(2183), Dimension(3082)])

feature_cols = trainNp.shape[1] # 3072
_, num_labels=labels_one_hot.get_shape().as_list() # 10