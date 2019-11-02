import numpy as np
import scipy 
from scipy import io as Imp
from numpy import linalg as LA

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
#%matplotlib inline
from numpy.linalg import norm
import warnings
warnings.filterwarnings("ignore")

from scipy.io import loadmat

#Question 1
Q1_train_faces = loadmat('faces.mat')['train_faces']
Q1_train_nonfaces = loadmat('faces.mat')['train_nonfaces']
Q1_test_faces = loadmat('faces.mat')['test_faces']
Q1_test_nonfaces = loadmat('faces.mat')['test_nonfaces']

#Getting the size of each data set
#print(np.size(Q1_test_faces))
#print(np.size(Q1_test_nonfaces))

#876869/361 = 2429 for each type of image for train
#170392/361 = 472 for each type of image for train

#initialize the training data and the training set
#Note I label face as 0 and no face as 1
Q1_train_dat = np.concatenate((Q1_train_faces, Q1_train_nonfaces), axis = 0)
Q1_train_y =  np.concatenate(( np.zeros(2429), np.full(2429,1)), axis = 0) 
Q1_test_dat = np.concatenate((Q1_test_faces, Q1_test_nonfaces), axis = 0)
Q1_test_y =  np.concatenate(( np.zeros(472), np.full(472,1)), axis = 0) 

#Not a good place to initialize
beta_hat = np.zeros(361)

#np.matmul(images, beta_hat)

# Speaking from experience, less painful code = happier TA :D
# Q1_Newton_Rapson: Performing Newton Rapson on Logistic Regression parameters
# Ibeta: Initialize beta vector
# x_i: data matrix
# y_i: result vector
# iter: number of iteration

def Q1_Newton_Rapson(Ibeta, x_i, y_i,itera):
    for x in range(0,itera):
        betax = np.matmul(x_i,Ibeta)
        pi = np.exp(betax)/(1+np.exp(betax))
        pip = np.multiply(np.subtract(1,pi), pi)
        W = np.diag(pip)
        
        #Makes the code somewhat readable
        XWX_inv = LA.inv(np.matmul(np.matmul(np.transpose(x_i),W),x_i))
        delta_beta = np.matmul(np.matmul(XWX_inv,np.transpose(x_i)), np.subtract(y_i, pi))
        Ibeta = np.add(Ibeta , delta_beta)
        #XD
    return Ibeta

#adding to the interception model, delete those 3 lines for non intercept model
Q1_train_dat = np.insert(Q1_train_dat,0, 1,axis = 1)
Q1_test_dat = np.insert(Q1_test_dat,0, 1,axis = 1)
beta_hat = np.insert(beta_hat,0, 1)

#Calculating Beta_hat and displaying it
beta_hat = Q1_Newton_Rapson(beta_hat,Q1_train_dat,Q1_train_y, 100)

print("Beta_hat_Matrix")
print(beta_hat)

Q1_train_betahatx = np.matmul(Q1_train_dat,beta_hat)
Q1_train_pi = np.exp(Q1_train_betahatx)/(1+np.exp(Q1_train_betahatx))

# Again, Less painful code = happier TA :D
Q1_train_result =Q1_train_pi>np.full((4858,),0.5)
Q1_train_error =Q1_train_result!=Q1_train_y


print("Training error: {}".format(sum(Q1_train_error)/4858))

#print(beta_hat)
Q1_test_betahatx = np.matmul(Q1_test_dat,beta_hat)
Q1_test_pi = np.exp(Q1_test_betahatx)/(1+np.exp(Q1_test_betahatx))

# Again, Less painful code = happier TA :D
Q1_test_result = Q1_test_pi>np.full((944,),0.5)
Q1_test_error = Q1_test_result!=Q1_test_y


print("Testing error: {}".format(sum(Q1_test_error)/944))

#Sanity check on the 