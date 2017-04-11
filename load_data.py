# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:03:19 2017

@author: sliu
read files with differernt categories from one folder
"""

import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np

direc = '/home/nw/sliu/Localdisk/github/cnn_hc-master/training_dataset/' 
# end with '/'
#bdirec = os.getcwd() # Get current working directory

category_keys = next(os.walk(direc))[1]
num_class  = len(category_keys)
class_values = range(num_class)
class_values = class_values
class_dict = dict(zip(category_keys,class_values))

#------------initial training data -------------
input_channel = 3
input_width   = 32
input_height  = 32
training_inputs  = np.empty( shape=(0,input_channel, input_width, input_height) )
training_results = np.empty( shape=(0, num_class) )

#-------------------------------------------
isamp = 0 #isamp for index of samples
class_labels = np.zeros(num_class)
#----------------
for name_class in category_keys:
    
    ipath  = direc + name_class 
    iclass = class_dict(name_class) 
    
    extension = '.png' # Select your file delimiter

    file_dict = {} # Create an empty dict

    # Select only files with the ext extension
    img_files = [i for i in os.listdir(direc) if os.path.splitext(i)[1] == extension]

    # Iterate over your txt files
    for ifile in img_files:
    # Open them and assign them to file_dict
        img    = load_img(os.path.join(direc,ifile))  #
        marray = img_to_array(img)  # this is a Numpy array with shape (3, 32, 32)
        training_inputs  = np.append(training_inputs, [marray],axis=0)
        
        class_labels[iclass] = 1.0  
        training_results     = np.append(training_results,[class_labels],axis=0)
        
        if len(training_results) != len(training_inputs):
            print('two arrays have different length')
            exit(1)
            
    print "class:",name_class,"has ",len(training_results)," samples."

# Iterate over your dict and print the key/val pairs.
print('Data loading over')
    

def get_immediate_subdirectories(main_dir):
    return [name for name in os.listdir(main_dir)
            if os.path.isdir(os.path.join(main_dir, name))]