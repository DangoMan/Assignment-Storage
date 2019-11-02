import numpy as np
import scipy 
from scipy import io as Imp
from numpy import linalg as LA

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
#%matplotlib inline
from numpy.linalg import norm

'''
import warnings
warnings.filterwarnings("ignore")
'''


from scipy.io import loadmat
import os
import tensorflow

'''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=""
'''


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

from tensorflow.python.client import device_lib

import pandas as pd 

from sklearn.model_selection import train_test_split

import csv

Testdat = pd.read_csv("test_data.csv",header=0)


Test_dat_x = np.array(Testdat.drop(columns=['ids']))


#Normalize data
Test_dat_x_normal = (Test_dat_x/255).astype('float32')
Test_dat_x_normal = np.reshape(Test_dat_x_normal,[Test_dat_x.shape[0],28,28,1])




#Taking some Vaildation data

input_shape = [1,784]
Kersize_Layer1 = ()

Classes = 10
epochs_cyc = 50
Batchsize = 100
'''
Train_dat_x_normal = np.array(Train_dat_x_normal)
Train_dat_y = np.array(Train_dat_y)
Vaid_x = np.array(Vaid_x)
Vaid_y = np.array(Vaid_y)
'''

model = load_model('CNN_model5Hyp.h5')


Score =  model.predict(Test_dat_x_normal, verbose=0)
print(Score[0])

out_lst = [['ids','label']]

for x in range(0,10000):
    lst = Score[x].tolist()
    Score_max = lst.index(max(lst))
    out_lst.append([x,Score_max])



  
myFile = open('Output1.csv', 'w')  

with myFile:  
    writer = csv.writer(myFile)
    writer.writerows(out_lst)

        

