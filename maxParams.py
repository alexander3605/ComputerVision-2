# Import libraries
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers import Input,Dense, Dropout, Flatten
import pydot
import graphviz


# GET DATA
from reader import get_images
(x_train, y_train), (x_test, y_test) = get_images()
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# DEFINE PARAMETERS
nInput = [10]
nHidden = [200]
lossFunct = ["categorical_crossentropy"]
opt = ["adam"]
metric = ["accuracy"]
batchSize = [150]
epoch = [10]

# DEFINE DATA STRUCTURE TO HOLD OPTIMAL COMBINATION
bestParams = [None, None, None, None, None, None, None]
bestScenario = (bestParams, 0)

#COUNT TOTAL NUMBER OF ITERATIONS
iterationTotal = len(nInput) * len(nHidden) * len(lossFunct) * len(opt) * len(metric) * len(batchSize) * len(epoch)

# ITERATE THROUGH ALL PARAMS COMBINATIONS
iterationCount = 0
for nI in nInput:
	for hH in nHidden:
		for lF in lossFunction:
			for oP in opt:
				for mT in metric:
					for bS in batchSize:
						for eP in epoch:
							
							iterationCount += 1
							print("Testing combination", iterationCount, "of", iterationTotal)
							# BUILD MODEL
							model = keras.models.Sequential()
							model.add(Dense(nI, input_shape=(784,), activation=tf.nn.relu))
							model.add(Dense(nH, activation=tf.nn.relu))
							model.add(Dense(10, activation=tf.nn.softmax))
							model.compile(loss=lF, optimizer=oP, metrics=[mT])
							# TRAIN MODEL
							batch_size = bS
							epochs = eP
							model.fit(x_train, y_train, batch_size, epochs)
							# EVALUATE ACCURACY
							score = model.evaluate(x_test, y_test)
							# COMPARE WITH BEST SOLUTION SO FAR
							if score[1] > bestScenario[1]:
								bestParams = [nI, nH, lF, oP, mT, bS, eP]
								bestScenario = (bestParams, score[1])
								print("======> Found better combination, accuracy: ", score[1])

# TELL USER BEST COMBINATION
print()
print("The best performance of", bestScenario[1], "is acheived with the parameters:")
print(bestScenario[0])
