import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers.core import Flatten
from keras.preprocessing.image import flip_axis
from keras.layers import Input, Lambda, Convolution2D, Dense, Dropout,Cropping2D
from keras.models import Model, model_from_json,Sequential
from scipy import misc
import random
import json
import csv
import os
import matplotlib.pyplot as plt
#import cv2
import time
###get_ipython().magic('matplotlib inline')

ImgDir=r'C:\Users\Siddarth\Desktop\Udacity\CarND-Behavioral-Cloning-P3-master\windows_sim\TrainingData\IMG\\'
LogFile=r'C:/Users/Siddarth/Desktop/Udacity/CarND-Behavioral-Cloning-P3-master/windows_sim/TrainingData/driving_log.csv'

UdacityDir=r'C:\Users\Siddarth\Desktop\Udacity\CarND-Behavioral-Cloning-P3-master\windows_sim\data\\'
UdacityLogFile=r'C:\Users\Siddarth\Desktop\Udacity\CarND-Behavioral-Cloning-P3-master\windows_sim\data\driving_log.csv'

X_train=np.load("X_train.npy")
y_train=np.load("y_train.npy")

print (X_train.shape,y_train.shape)


model=Sequential()
model.add(Lambda(lambda x: x/255.-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.3,shuffle=True)
model.save('model.h5')
