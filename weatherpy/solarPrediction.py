from __future__ import print_function
import csv
import pandas as pd
import tensorflow as tf
import numpy as np
import math
import tflearn
from datetime import datetime
from tflearn.datasets import titanic
from tflearn.data_utils import load_csv, to_categorical

# The file containing the weather samples (including the column header)
WEATHER_SAMPLE_FILE = 'weather.csv'
data, labels = load_csv(WEATHER_SAMPLE_FILE, target_column=11, columns_to_ignore=[0])

TrainingSetFeatures = data
TrainingSetLabels = labels

def preprocessor(data):
	copyData = np.zeros((len(data), 12))
	for i in range(len(data)):
		sample = data[i]
		# filter out any samples that are way off.
		if(float(sample[9])< 25.0 or float(sample[8]) < 25.0):
			continue
		#grab the date element
		dayStr = sample[0]
		#print("datetime",sample[0])
		dayOfYear = datetime.strptime(dayStr, "%m/%d/%Y").timetuple().tm_yday
		hours = int(sample[1])
		hourVectorReal = math.cos(2*math.pi * (hours/24))
		#print("hour",hourVectorReal)
		hourVectorImg = math.sin(2*math.pi * (hours/24))		
		dayVectorReal = math.cos(2*math.pi * (dayOfYear/365))
		dayVectorImg = math.sin(2*math.pi * (dayOfYear/365))
		copyData[i][0] = hourVectorReal 
		copyData[i][1] = hourVectorImg 
		copyData[i][2] = dayVectorReal
		copyData[i][3] = dayVectorImg
		#cloud coverage
		copyData[i][4] = sample[2]
		#visibility
		copyData[i][5] = sample[3]
		#tempreture
		copyData[i][6] = sample[4]
		#dew point
		copyData[i][7] = sample[5]
		#relative humidity
		copyData[i][8] = sample[6]
		#wind speed
		copyData[i][9] = sample[7]
		#station
		copyData[i][10] = sample[8]
		#altimeter

		copyData[i][11] = sample[9]
	return copyData

def categorizeLabels(labels):
	for i in range(len(labels)):
		evSample = float(labels[i])
		if evSample > 4000:
			labels[i] = 8
		elif evSample > 3500:
			labels[i] = 7
		elif evSample > 3000:
			labels[i] = 6
		elif evSample > 2500:
			labels[i] = 5
		
		if evSample > 2000:
			labels[i] = 4
		elif evSample > 1500:
			labels[i] = 3
		elif evSample > 1000:
			labels[i] = 2
		elif evSample > 500:
			labels[i] = 1
		else:
			labels[i] = 0
			

TrainingSetFeatures = preprocessor(TrainingSetFeatures)
categorizeLabels(TrainingSetLabels)
TrainingSetLabels = to_categorical(TrainingSetLabels, 9)

#create a test set from the number of samples and traning set
net = tflearn.input_data(shape=[None, 12])
net =tflearn.fully_connected(net,32)
net = tflearn.highway(net, 32, activation="LeakyReLu",name="ReLuLayer")
net =tflearn.fully_connected(net,32)

net = tflearn.fully_connected(net,  9, activation="softmax")

net = tflearn.regression(net,learning_rate = 0.005)
#adam = tflearn.Optimizer()
#net = tflearn.regression(net, learning_rate=0.001, optimizer=adam)
# Define model
model = tflearn.DNN(net, clip_gradients=1.0, tensorboard_verbose=3, tensorboard_dir='./tmp/weather1.log')

# Start training (apply gradient descent algorithm)
model.fit(TrainingSetFeatures, TrainingSetLabels, n_epoch=10, batch_size=40, show_metric=True)





