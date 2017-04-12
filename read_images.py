# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:03:19 2017

@author: sliu
read files with differernt categories from one folder
"""

import os
from keras import backend as K 
from keras.preprocessing.image import img_to_array, load_img
K.set_image_dim_ordering('th')

import numpy as np
import pickle 

def main():
    #---------------setting initial value of parameters-------------
    input_channel = 3
    input_width   = 32
    input_height  = 32
    
    direc   = '/home/nw/sliu/Localdisk/github/cnn_hc-master/training_dataset/' # end with '/'
    pklfile ='training_data.pkl' 
    read_images(direc,pklfile,input_channel,input_width,input_height)
    
    
    direc_test   = '/home/nw/sliu/Localdisk/github/cnn_hc-master/testing_dataset/' # end with '/'
    pklfile_test ='testing_data.pkl' 
    
    read_images(direc_test,pklfile_test,input_channel,input_width,input_height)


def read_images(direc,pklfile,input_channel,input_width,input_height):
    category_keys = next(os.walk(direc))[1] #subdirectoy
    num_class     = len(category_keys)
    class_values  = range(num_class)
    class_dict    = dict(zip(category_keys,class_values))
    
    #------------initial arrays for training data -------------
    training_inputs  = np.empty( shape=(0,input_channel, input_width, input_height) )
    training_results = np.empty( shape=(0, num_class) )
    
    
    #-------------------------------------------
    class_labels = np.zeros(num_class)
    extension = '.png' # Select your file delimiter
    #----------------
    for name_class in category_keys:
        
        print '--->start to read data from class:',name_class    
        ipath  = direc + name_class 
        iclass = class_dict[name_class] 
    
        # Select only files with the ext extension
        img_files = [i for i in os.listdir(ipath) if os.path.splitext(i)[1] == extension]
    
        # Iterate over your txt files
        for ifile in img_files:
        # Open them and assign them to file_dict
    
            img    = load_img(os.path.join(ipath,ifile))  #
            marray = img_to_array(img)  # this is a Numpy array with shape (3, 32, 32)
            #print 'marray shape:',marray.shape        
            
            training_inputs      = np.append(training_inputs,[marray],axis=0)
            
            class_labels[iclass] = 1.0  
            training_results     = np.append(training_results,[class_labels],axis=0)
         
            if len(training_results) != len(training_inputs):
                print('two arrays have different length')
                exit(1)
                
                
        print "class:",name_class,"has ",len(img_files)," samples."
    
    
    #-----------------save with pickle----------------
    with open(pklfile,'wb') as pf:
         pickle.dump([class_dict,input_channel, input_width, input_height,training_inputs,training_results],pf)
    #-----------load data using below codes--------------
    """
    with open(pklfile,'rb') as pf:
         [class_dict,input_channel, input_width, input_height,training_inputs,training_results]=pickle.load(pf)
    
    """
     
    #--------------successfule message----------------
    print 'There are ',num_class,' categories and ',len(training_results),'samples' 
    print('....Data loading over and saved successfully....:):)')


#---------------------------------------------------------------------------------------

                
if __name__ == '__main__':
    main()