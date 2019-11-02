#!/usr/bin/env python
# coding: utf-8

# In[24]:


#Handling the import
#Sorry, my hard drive got wipe on friday, so my code might seem a bit panicky

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
import random
from sklearn.cluster import KMeans


# In[25]:


#importing the data
#Question 1
Ion_dat_x = loadmat('Ion.mat')['X']
Ion_dat_y = loadmat('Ion.mat')['y']
print(np.shape(Ion_dat_x))
print(np.shape(Ion_dat_y))


# In[26]:


#Input:
# cluster_pts: training list
# Train_lab: train label
# Kclust: Number of K cluster points
def RBF_Train_max(cluster_pts,Train_lab,Kclust):
    
    #obtaining the list of means
    mean_lst = clusterpt(cluster_pts,Kclust)
    
    def red_gen(X_mat):
        red_mat = []
        len_lst = np.shape(X_mat)[1]
        for x in range(0,Kclust):
            red_vec = []
            for y in range(0,np.shape(X_mat)[0]) :
                red_vec.append(np.exp(-np.sum(np.square(X_mat[y] - mean_lst[x]))/20))
            red_mat.append(red_vec)
        return np.asarray(red_mat)

    phi = red_gen(cluster_pts)
    weight = np.matmul(np.matmul(LA.inv(np.matmul(phi,phi.T)), phi), Train_lab)


    return red_gen, weight


#clustering algo:
#Input: list of points, number of clusters
#Outputs: center point of each cluster
def clusterpt(lst, m):
    pred = KMeans(n_clusters = m, random_state = 17).fit(lst)
    return pred.cluster_centers_
    


# In[27]:


#Randomize the list (Since the last few entry tends to be all zeros)
random.seed(34)
Ion_dat_x = Ion_dat_x.T
combined= list(zip (Ion_dat_x,Ion_dat_y))
random.shuffle(combined)
Ion_dat_x[:], Ion_dat_y[:] = zip (*combined)


# In[28]:


#k cluster number of clusters
m = 4


# In[29]:


#Standard cross validation
#There is 351 points: use 80-20 train test, 281 train, 70 test
def Crossval(clustdim):
    Ion_dat_CV_x_train = Ion_dat_x[:281,:]
    Ion_dat_CV_y_train = Ion_dat_y[:281]
    Ion_dat_CV_x_test = Ion_dat_x[281:,:]
    Ion_dat_CV_y_test = Ion_dat_y[281:]


    CV_redgen, CV_weight = RBF_Train_max(Ion_dat_CV_x_train, Ion_dat_CV_y_train , clustdim)

    CV_adj = CV_redgen(Ion_dat_CV_x_train)
    CV_Pred_lst = np.matmul(CV_adj.T,CV_weight)
    CV_Train_Pred=sum((CV_Pred_lst>0.5)== Ion_dat_CV_y_train)/281

    CV_Test_adj = CV_redgen(Ion_dat_CV_x_test)
    CV_Test_Pred_lst = np.matmul(CV_Test_adj.T,CV_weight)
    CV_Test_Pred=sum((CV_Test_Pred_lst>0.5)== Ion_dat_CV_y_test)/70
    
    return CV_Train_Pred, CV_Test_Pred
    
CV_train_err, CV_Test_err = Crossval(m)


# In[30]:


#Leave one out cross validation
def LOOvali(clustdim):
    LOO_train_err = 0
    LOO_test_err = 0
    
    for x in range(0,351):
        Ion_dat_LOO_x_test = Ion_dat_x[x:x+1,:]
        Ion_dat_LOO_y_test = Ion_dat_y[x:x+1]
        Ion_dat_LOO_x_train = np.delete(Ion_dat_x,x,0)
        Ion_dat_LOO_y_train = np.delete(Ion_dat_y,x)
    
        LOO_redgen, LOO_weight = RBF_Train_max(Ion_dat_LOO_x_train, Ion_dat_LOO_y_train , clustdim)    
        
        LOO_adj = LOO_redgen(Ion_dat_LOO_x_train)
        LOO_Pred_lst = np.matmul(LOO_adj.T,LOO_weight)
        LOO_train_err+=sum((LOO_Pred_lst>0.5)== Ion_dat_LOO_y_train)/350
        
        LOO_Test_adj = LOO_redgen(Ion_dat_LOO_x_test)
        LOO_Test_Pred_lst = np.matmul(LOO_Test_adj.T,LOO_weight)
        LOO_test_err+=sum((LOO_Test_Pred_lst>0.5)== Ion_dat_LOO_y_test)
        
    return LOO_train_err/351, LOO_test_err/351

LOO_train_err, LOO_test_err = LOOvali(m)

print(LOO_train_err)
print(LOO_test_err)


# In[34]:


#CLOO
def CLOOvali(clustdim):
    CLOO_redgen, CLOO_weight = RBF_Train_max(Ion_dat_x, Ion_dat_y , clustdim)
    
    CLOO_adj = CLOO_redgen(Ion_dat_x)
    CLOO_Pred_lst = np.matmul(CLOO_adj.T,CLOO_weight)
    CLOO_train_err=sum((CLOO_Pred_lst>0.5)== Ion_dat_y)/351

    #Calculating Hat matrix
    Hat_mat = np.matmul(np.matmul(CLOO_adj.T, LA.inv(np.matmul(CLOO_adj,CLOO_adj.T))),CLOO_adj)
    Hatdiag = np.diag(Hat_mat)
    
    sum_test_err = 0
    
    for x in range(0,351):
        pred_x = ((CLOO_Pred_lst[x]>0.5)== Ion_dat_y[x])/(1-Hatdiag[x])
        sum_test_err += np.floor(pred_x)
    
    return CLOO_train_err, sum_test_err/351

CLOO_train_err, CLOO_test_err = CLOOvali(m)

    


# In[36]:


#Finding the best clustering 

for m in range(1,31):
    CV_train_err, CV_Test_err = Crossval(m)
    LOO_train_err, LOO_test_err = LOOvali(m)
    CLOO_train_err, CLOO_test_err = CLOOvali(m)
    
    print(m)
    print("Method \t Training Error \t Testing Error")
    print("CV \t {} \t {}".format(1- CV_train_err, 1-CV_Test_err))
    print("LOO \t {} \t {}".format(1-LOO_train_err, 1-LOO_test_err))
    print("CLOO \t {} \t {}".format(1- CLOO_train_err,1- CLOO_test_err))

