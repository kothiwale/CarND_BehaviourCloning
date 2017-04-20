
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras.layers.core import Flatten
from keras.preprocessing.image import flip_axis
from keras.layers import Input, Lambda, Convolution2D, Dense, Dropout,Cropping2D
from keras.models import Model, model_from_json,Sequential
import random
import csv
import os
import cv2
import time
get_ipython().magic('matplotlib inline')

ImgDir=r'C:\Users\User\Desktop\CarND_BehaviorCloning\windows_sim\TrainingData\IMG\\'
LogFile=r'C:\Users\User\Desktop\CarND_BehaviorCloning\windows_sim/TrainingData/driving_log.csv'

UdacityDir=r'C:\Users\User\Desktop\CarND_BehaviorCloning\windows_sim\data\\'
UdacityLogFile=r'C:\Users\User\Desktop\CarND_BehaviorCloning\windows_sim\data\driving_log.csv'


def convertImage(image):
    ''' This funcion is used to pre-process the images. White, red, yellow and red lane markings are extractes and other parts of image
     are pushed to the backgrounf'''
    
    cutImage=image#[60:140,:]
    
    line_image=np.copy(cutImage)
    
    hsv=cv2.cvtColor(line_image,cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([23,41,133])
    upper_yellow = np.array([40,150,255])

    lower_red = np.array([150,0,0])
    upper_red = np.array([180,200,255])
    
    lower_black = np.array([0,0,0])
    upper_black = np.array([120,120,50])
    
    sensitivity=40
    lower_white = np.array([0,0,255-sensitivity], dtype=np.uint8)
    upper_white = np.array([255,sensitivity,255], dtype=np.uint8)
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red = cv2.inRange(hsv,lower_red,upper_red)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    masked_black = cv2.addWeighted(cutImage, 0.5,cv2.cvtColor(mask_black, cv2.COLOR_GRAY2RGB), 1, 0)
    masked_red= cv2.addWeighted(masked_black, 0.5,cv2.cvtColor(mask_red, cv2.COLOR_GRAY2RGB), 1, 0)
    masked_yellow= cv2.addWeighted(masked_red, 0.5,cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2RGB), 1, 0)
    masked_white= cv2.addWeighted(masked_yellow, 0.5,cv2.cvtColor(mask_white, cv2.COLOR_GRAY2RGB), 1, 0)
    
    mask= masked_white
    
    return mask

### Reading, processing and appending the image to the array for self generated image

lines=[]
measurements=[]
images=[]

with open(LogFile) as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)


for line in lines:
    for i in range (3):
        source_path=line[i]
        if i==0:
            correction=0
        if i==1:
            correction=0.2
        if i==2:
            correction=-0.2
        filename=source_path.split('\\')[-1]
        ##Dir=source_path.split('\\')[:-1]
        current_path=ImgDir+filename
        image=plt.imread(current_path)
        #image=convertImage(image)
        images.append(image)
        measurement=float(line[3])+correction
        measurements.append(measurement)
        if (measurement<-0.4 or measurement>0.4):
            flipImg=np.fliplr(image)
            images.append(flipImg)
            measurements.append(-measurement)
    
X_train=np.array(images)
y_train=np.array(measurements)

print (X_train.shape,y_train.shape)


# The Model starts here

model=Sequential()
model.add(Lambda(lambda x: x/255.-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


# Compiling and training the model. The model is saved as 'model.h5'

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.3,shuffle=True,batch_size=128)
model.save('model.h5')

