from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import pickle
import numpy as np
#import json
from keras.models import model_from_json
from keras import backend as K ##https://github.com/fchollet/keras/issues/2681
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

K.set_image_dim_ordering('th')

#----------------------------saved model and parametres---
model_file = "model.json"
weights_file ="Class_4.h5"

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weights_file)
print("Loaded model from disk")

# evaluate loaded model on test data
sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
loaded_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


"""
############################## LOAD DATA #####################################
DATA_FILE = 'img_data.pkl'
NUM_CLASS = 4
NUM_TRAIN_IMG = 2550
NUM_TEST_IMG = 100
INPUT_CHANNEL = 3
INPUT_WIDTH = 90
INPUT_HEIGHT = 60
f = open(DATA_FILE, 'rb')

training_data = pickle.load(f)
training_inputs = np.reshape(training_data[0], (NUM_TRAIN_IMG, INPUT_CHANNEL, INPUT_WIDTH, INPUT_HEIGHT))
training_results = np.reshape(training_data[1], (NUM_TRAIN_IMG, NUM_CLASS))

eval_data = pickle.load(f)
X_test = np.reshape(eval_data[0], (NUM_TEST_IMG, INPUT_CHANNEL, INPUT_WIDTH, INPUT_HEIGHT))
y_test = np.reshape(eval_data[1], (NUM_TEST_IMG, NUM_CLASS))

f.close()
###############################################################################

"""

#--------------------------------------------------------------------------
NUM_CLASS = 4
Num_train_samples = [500,500,500,500]
Num_test_samples  = [500,500,500,500]
INPUT_CHANNEL = 3
INPUT_WIDTH   = 32
INPUT_HEIGHT  = 32
FrequencyLabels = ["cos128","cos133","cos138", "cos143"]
NUM_TRAIN_IMG     = np.sum(Num_train_samples)
NUM_TEST_IMG      = np.sum(Num_test_samples)

X_test = np.zeros((NUM_TEST_IMG, INPUT_CHANNEL, INPUT_WIDTH, INPUT_HEIGHT))
Y_test = np.zeros((NUM_TEST_IMG, NUM_CLASS))
#------------------------class one for test --------------
for j in xrange(0,len(FrequencyLabels)):
    labels = FrequencyLabels[j]
    class_folder = "training_dataset/"+labels+"/"
    index_starting = np.sum(Num_train_samples[0:j])
    for i in xrange(0, Num_train_samples[j]):
        index=1000+i # image name with fixed 4-number index
        imagepath = class_folder+labels+"_"+str(index)+".png"
        img = load_img(imagepath)  #
        marray =  img_to_array(img)  # this is a Numpy array with shape (3, 32, 32)
        ilabel =  int(i+index_starting)
        X_test[ilabel,]=marray
        Y_test[ilabel,j]=1.0
        
        
#--------------------------------------------------------------
score = loaded_model.evaluate(X_test, Y_test, batch_size=16)
#score = loaded_model.evaluate(X_test, Y_test, verbose=0)

#-------------------re-train from the previous model---------------

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


#probs = loaded_model.predict(X_test, batch_size=16)
#predictions = probs.argmax(axis=1)
