import numpy as np
import scipy 
from scipy import io as Imp
from numpy import linalg as LA

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
#%matplotlib inline
from numpy.linalg import norm
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

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

from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.models import load_model

from tensorflow.python.client import device_lib

import pandas as pd 

from sklearn.model_selection import train_test_split

Traindat = pd.read_csv("fashion-mnist_train.csv",header=0)


Train_dat_y = np.array(Traindat['label'])
Train_dat_x = np.array(Traindat.drop(columns=['label']))




Train_dat_x_new, Vaid_x, Train_dat_y_new, Vaid_y = train_test_split( Train_dat_x, Train_dat_y, test_size=0.2, random_state=17)

#Normalize data
Train_dat_x_normal = (Train_dat_x_new/255).astype('float32')
x_train = np.reshape(Train_dat_x_normal,[Train_dat_x_new.shape[0],28,28,1])
Vaid_x = (Vaid_x/255).astype('float32')
x_test = np.reshape(Vaid_x,[Vaid_x.shape[0],28,28,1])

#Convert this into One hot encoding
y_train =  np_utils.to_categorical(Train_dat_y_new,10)
y_test =  np_utils.to_categorical(Vaid_y,10)

#Making the model

Classes = 10
epochs_cyc = 50
Batchsize = 100

input_shape = (28, 28, 1)
def train_generator(x, y, batch_size, shift_fraction=0.):
    train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                       height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
    generator = train_datagen.flow(x, y, batch_size=500)
    while 1:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])

def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    return train_model, eval_model
    
def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))



# setting the hyper parameters
model, eval_model = CapsNet(input_shape=input_shape,
                                              n_class=10,
                                              routings=3)


model.compile(optimizer=optimizers.Adam(0.001),
              loss=[margin_loss, 'mse'],
              loss_weights=[1., 0.392],
              metrics={'capsnet': 'accuracy'})

checkpointer = keras.callbacks.ModelCheckpoint('logs/Capnet_model_check.h5', verbose=0, save_best_only=True, save_weights_only=True, period=1)

model.fit_generator(generator=train_generator(x_train, y_train, 500, 0.1),  epochs=epochs_cyc,
              validation_data=[[x_test, y_test], [y_test, x_test]],callbacks=[checkpointer])

model.save('Capnet_model2.h5')


