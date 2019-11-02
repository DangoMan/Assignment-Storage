import numpy as np
import scipy 
from scipy import io as Imp
from numpy import linalg as LA

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
#%matplotlib inline
from numpy.linalg import norm


from hyperas.distributions import uniform

from hyperas import optim
from hyperas.distributions import choice, uniform


'''
import warnings
warnings.filterwarnings("ignore")
'''


from scipy.io import loadmat
import os
import tensorflow


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=""



import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from hyperopt import Trials, STATUS_OK, tpe

from tensorflow.python.client import device_lib

import pandas as pd 

from sklearn.model_selection import train_test_split
def data():
    Traindat = pd.read_csv("fashion-mnist_train.csv",header=0)
    Testdat = pd.read_csv("test_data.csv",header=0)


    Train_dat_y = np.array(Traindat['label'])
    Train_dat_x = np.array(Traindat.drop(columns=['label']))
    Test_dat_y = np.array(Traindat['label'])
    Test_dat_x = np.array(Traindat.drop(columns=['label']))


    Train_dat_x_new, Vaid_x, Train_dat_y_new, Vaid_y = train_test_split( Train_dat_x, Train_dat_y, test_size=0.2, random_state=17)


    #Normalize data
    Train_dat_x_normal = (Train_dat_x_new/255).astype('float32')
    Train_dat_x_normal = np.reshape(Train_dat_x_normal,[Train_dat_x_new.shape[0],28,28,1])
    Vaid_x = (Vaid_x/255).astype('float32')
    Vaid_x = np.reshape(Vaid_x,[Vaid_x.shape[0],28,28,1])

    #Convert this into One hot encoding
    Train_dat_y_new =  np_utils.to_categorical(Train_dat_y_new,10)
    Vaid_y =  np_utils.to_categorical(Vaid_y,10)

    return Train_dat_x_normal, Train_dat_y_new, Vaid_x,Vaid_y



#Taking some Vaildation data
#np.random.seed(3417)
def create_model(Train_dat_x_normal, Train_dat_y_new, Vaid_x,Vaid_y):
    Kersize_Layer1 = ()

    Classes = 10
    epochs_cyc = 200
    Batchsize = 100

    model = Sequential()
    model.add(Conv2D({{choice([16,32,64,128])}}, kernel_size={{choice([(3,3),(4,4)])}}, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D({{choice([16,32,64,128])}}, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D({{choice([32,64,128])}}, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D({{choice([32,64,128])}}, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D({{choice([32,64,128])}}, kernel_size=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Conv2D({{choice([32,64,128])}}, kernel_size={{choice([(3,3),(4,4)])}}, activation='relu'))
    model.add(Conv2D({{choice([16,32,64,128])}}, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D({{choice([16,32,64,128])}},  kernel_size=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense(Classes, activation='softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    result = model.fit(Train_dat_x_normal, Train_dat_y_new, validation_data=(Vaid_x, Vaid_y), epochs=epochs_cyc, batch_size=Batchsize, verbose=2,validation_split=0.1)
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:'.format(validation_acc))
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

X_train, Y_train, X_test, Y_test = data()
best_run, best_model = optim.minimize(model=create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials())

print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)
best_model.save('CNN_model6Hyp.h5') 

