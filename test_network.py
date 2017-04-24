# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 23:45:00 2017

@author: hahayi
"""
#load images data
#method1:https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#method2:https://gurus.pyimagesearch.com/lesson-sample-running-a-pre-trained-network/#
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import pickle
import numpy as np
import json
from keras import backend as K ##https://github.com/fchollet/keras/issues/2681
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
K.set_image_dim_ordering('th')
#-----------------------------------------------------
NUM_CLASS = 4
Num_train_samples = [500,500,500,500]
Num_test_samples  = [500,500,500,500]
NUM_TRAIN_IMG     = np.sum(Num_train_samples)
NUM_TEST_IMG      = np.sum(Num_test_samples)

INPUT_CHANNEL = 3
INPUT_WIDTH   = 32
INPUT_HEIGHT  = 32
FrequencyLabels = ["cos128","cos133","cos138", "cos143"]

#-------------------load training dataset----------------------------------
training_inputs=np.zeros((NUM_TRAIN_IMG, INPUT_CHANNEL, INPUT_WIDTH, INPUT_HEIGHT))
training_results=np.zeros((NUM_TRAIN_IMG,NUM_CLASS))


"""
creat a list,loop over this list,including class name,sample number,folder path
"""

for j in xrange(0,len(FrequencyLabels)):
    labels = FrequencyLabels[j]
    class_folder = "training_dataset/"+labels+"/"
#---------------------class one-------------------
    index_starting = np.sum(Num_train_samples[0:j])
    for i in xrange(0, Num_train_samples[j]):
        index=1000+i # image name with fixed 4-number index
        imagepath = class_folder+labels+"_"+str(index)+".png"
        img = load_img(imagepath)  #
        marray = img_to_array(img)  # this is a Numpy array with shape (3, 32, 32)
        ilabel =  int(i+index_starting)
        training_inputs[ilabel,] = marray
        training_results[ilabel,j] = 1.0

#----------------------------------------------------------------
print "load testing data"
#-----------------------load testing dataset-------------------

X_test = np.zeros((NUM_TEST_IMG, INPUT_CHANNEL, INPUT_WIDTH, INPUT_HEIGHT))
Y_test = np.zeros((NUM_TEST_IMG, NUM_CLASS))
#------------------------class one for test --------------
for j in xrange(0,len(FrequencyLabels)):
    labels = FrequencyLabels[j]
    class_folder = "testing_dataset/"+labels+"/"
    index_starting = np.sum(Num_train_samples[0:j])
    for i in xrange(0, Num_train_samples[j]):
        index=1000+i # image name with fixed 4-number index
        imagepath = class_folder+labels+"_"+str(index)+".png"
        img = load_img(imagepath)  #
        marray =  img_to_array(img)  # this is a Numpy array with shape (3, 32, 32)
        ilabel =  int(i+index_starting)
        X_test[ilabel,]=marray
        Y_test[ilabel,j]=1.0

#---------------------pickle format data ----------------------
"""

"""
#--------------------------------------------------------------


############################ DESIGN MODEL #####################################
model = Sequential()

# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(16, 3, 3, border_mode='valid', input_shape=(INPUT_CHANNEL, INPUT_WIDTH, INPUT_HEIGHT)))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256)) # overflow error

model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(NUM_CLASS))
model.add(Activation('softmax'))
#################################################################################

############################ TRAINING ###########################################
sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# starts training
model.fit(training_inputs, training_results, nb_epoch=50, batch_size=32)
#################################################################################

############################ SAVE MODEL #########################################
json_string = model.to_json()
with open("model.json", "w") as outfile:
    outfile.write(json_string)

SAVED_WEIGHTS = "Class_4.h5"
model.save_weights(SAVED_WEIGHTS) # save weights
#################################################################################

############################ EVALUATION #########################################
score = model.evaluate(X_test, Y_test, batch_size=16)

print score


