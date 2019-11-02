#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Handling the import

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

#Importing the Optimizer 

import sys
sys.path.append("C:\Program Files (x86)\Python\lib\site-packages\cvxopt")

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


# In[12]:


#Importing data
Linear_X =  loadmat('linear_new.mat')['X']
Linear_Y =  loadmat('linear_new.mat')['y']
NoiLinear_X =  loadmat('noisylinear_new_1.mat')['X']
NoiLinear_Y =  loadmat('noisylinear_new_1.mat')['y']
Quad_X =  loadmat('quadratic_new.mat')['X']
Quad_Y =  loadmat('quadratic_new.mat')['y']

#Linear: 51 points: 40 train, 11 test
Linear_test_X = Linear_X[:,40:]
Linear_test_Y = Linear_Y[40:]
Linear_X = Linear_X[:,:40]
Linear_Y = Linear_Y[:40]

#Noisy Linear: 101 points: 80 train, 21 test
NoiLinear_Test_X = NoiLinear_X[:,80:]
NoiLinear_Test_Y = NoiLinear_Y[80:]
NoiLinear_X = NoiLinear_X[:,:80]
NoiLinear_Y = NoiLinear_Y[:80]

#Quadratic: 101 points: 80 train, 21 test
Quad_Test_X = Quad_X[:,80:]
Quad_Test_Y = Quad_Y[80:]
Quad_X = Quad_X[:,:80]
Quad_Y = Quad_Y[:80]


# In[4]:


#2a 
def HardMarg(X,y):
    #Note the problem is to:
    #max \sigma_i alpha - 0.5 \sigma_{i,j} alpha_i alpha_j y_i y_j x^T_i x_j
    #subject to alpha_i \geq 0, \sigma \alpha_i y_i = 0
    #Form of qt solver (http://cvxopt.org/userguide/coneprog.html): 
    # min 0.5 (alpha^T P alpha) + q alpha
    # subject to G alpha < h
    # A alpha = b
    # In this case (mutiply by -1):
    # P_ij = y_i y_j x^T_i x_j, P = y^T X^T X y
    # q = -1
    # G = -1
    # h = 0
    # a = y
    # b = 0
    d = np.shape(X)[1]
    y = y.astype(np.double)
    
    P = cvxopt_matrix(np.matmul(y, y.T)*np.matmul(X.T, X))
    q = cvxopt_matrix(-np.ones((d,1)))
    G = cvxopt_matrix(-np.eye(d))
    h = cvxopt_matrix(np.zeros(d))
    a = cvxopt_matrix(y.T)
    b = cvxopt_matrix(np.zeros(1))
    
    alpha = np.array(cvxopt_solvers.qp(P, q, G, h, a, b,kktsolver='ldl', options={'kktreg':1e-9, 'show_progress':False})['x'])
    
    beta = np.sum((alpha * y).T * X,axis=1)
    count = 0
    while(alpha[count]<=1e-5):
            count += 1
            
    beta_0 = y[count] - np.matmul(beta,X[:,count:count+1])
    
    return beta, beta_0
    
Hardbeta, Hardbeta_0 = HardMarg(NoiLinear_X,NoiLinear_Y)

print(Hardbeta)
print(Hardbeta_0)
    


# In[5]:


#2b
def SoftMarg(X,y,gamma):
    #Same as above, but 
    # 0 < alpha < gamma
    # or 
    # [alpha, -alpha] < [0,gamma]
    d = np.shape(X)[1]
    y = y.astype(np.double)
    X_dash = y *  X.T
    
    #P = cvxopt_matrix(np.matmul(y, y.T)*np.matmul(X.T, X))
    P = cvxopt_matrix(np.dot(X_dash,X_dash.T))
    q = cvxopt_matrix(-np.ones((d,1)))
    G = cvxopt_matrix(np.vstack((-np.eye(d), np.eye(d))))
    h = cvxopt_matrix(np.hstack((np.zeros(d), np.ones(d) * gamma)))
    a = cvxopt_matrix(y.T)
    b = cvxopt_matrix(np.zeros(1))
    
    alpha = np.array(cvxopt_solvers.qp(P, q, G, h, a, b,kktsolver='ldl', options={'kktreg':1e-9, 'show_progress':False})['x'])
    
    beta = np.sum((alpha * y).T * X,axis=1)
    count = 0
    while(alpha[count]<=1e-5):
            count += 1
            
    beta_0 = y[count] - np.matmul(beta,X[:,count:count+1])
    
    return beta, beta_0
    


# In[6]:


#2c)
def sign(x):
    if(x >= 0):
        return 1
    else :
        return -1

def classify(Xtest,beta,beta0):
    d = np.matmul(Xtest.T,beta) + np.ones(np.shape(Xtest)[1])*beta0
    return list(map(sign, d))


# In[7]:


#Prep for plot
def divlst(X,y):
    lstp = []
    lstn = []
    for i in range(0,np.shape(y)[0]):
        if(y[i] == 1 ):
            lstp.append(X[:,i:i+1])
        else:
            lstn.append(X[:,i:i+1])
    return lstp, lstn

def linegen(max, min, beta,beta0):
    xlst = [min,max]
    ylst = [-(min * beta[0] + beta0)/beta[1], -(max * beta[0] + beta0)/beta[1]]
    return(xlst,ylst)
    


# In[14]:


#2d)
#Linear 
#Red: Training points with y = 1
#Blue: Trainging points with y = -1
#Magenta : Testing points with y = 1
#Black : Testing points with y = -1
#Green: Hard Margin
#Yellow: Soft Margin
Linear_X_p, Linear_X_n = divlst(Linear_X,Linear_Y)
for x in Linear_X_p:
    plt.plot(x[0],x[1], c='r', marker = '.')
for x in Linear_X_n:
    plt.plot(x[0],x[1], c='b', marker = '.')
    
Linear_X_test_p, Linear_X_test_n = divlst(Linear_test_X,Linear_test_Y)
for x in Linear_X_test_p:
    plt.plot(x[0],x[1], c='m', marker = '.')
for x in Linear_X_test_n:
    plt.plot(x[0],x[1], c='k', marker = '.')

#Hard Marg
Lin_beta_H, Lin_beta0_H = HardMarg(Linear_X,Linear_Y)
Lin_beta_xlst_H,Lin_beta_ylst_H =  linegen(0,1,Lin_beta_H,Lin_beta0_H)
plt.plot(Lin_beta_xlst_H,Lin_beta_ylst_H, c='g')

Lin_beta_S, Lin_beta0_S = SoftMarg(Linear_X,Linear_Y,0.5)
Lin_beta_xlst_S,Lin_beta_ylst_S =  linegen(0,1,Lin_beta_S,Lin_beta0_S)
plt.plot(Lin_beta_xlst_S,Lin_beta_ylst_S, c='y')

lst_LinH = classify(Linear_X, Lin_beta_H, Lin_beta0_H) != Linear_Y.T
Train_err_LinH = np.sum(lst_LinH,axis=1)/np.shape(Linear_Y)[0]
lst_LinH_Test = classify(Linear_test_X, Lin_beta_H, Lin_beta0_H) != Linear_test_Y.T
Train_err_LinH_test = np.sum(lst_LinH_Test,axis=1)/np.shape(Linear_test_Y)[0]

print("Linear data")
print("Traing error Hard margin: {}".format(Train_err_LinH))
print("Testing error Hard margin: {}".format(Train_err_LinH_test))

lst_LinS = classify(Linear_X, Lin_beta_S, Lin_beta0_S) != Linear_Y.T
Train_err_LinS = np.sum(lst_LinS,axis=1)/np.shape(Linear_Y)[0]
lst_LinS_Test = classify(Linear_test_X, Lin_beta_S, Lin_beta0_S) != Linear_test_Y.T
Train_err_LinS_test = np.sum(lst_LinS_Test,axis=1)/np.shape(Linear_test_Y)[0]

print("Traing error Soft margin: {}".format(Train_err_LinS))
print("Testing error Hard margin: {}".format(Train_err_LinS_test))


# In[18]:


#Noisy Linear
#Red: Training points with y = 1
#Blue: Trainging points with y = -1
#Magenta : Testing points with y = 1
#Black : Testing points with y = -1
#Green: Hard Margin
#Yellow: Soft Margin
NoiLinear_X_p, NoiLinear_X_n = divlst(NoiLinear_X,NoiLinear_Y)
for x in NoiLinear_X_p:
    plt.plot(x[0],x[1], c='r', marker = '.')
for x in NoiLinear_X_n:
    plt.plot(x[0],x[1], c='b', marker = '.')

NoiLinear_X_test_p, NoiLinear_X_test_n = divlst(NoiLinear_Test_X,NoiLinear_Test_Y)
for x in NoiLinear_X_test_p:
    plt.plot(x[0],x[1], c='m', marker = '.')
for x in NoiLinear_X_test_n:
    plt.plot(x[0],x[1], c='k', marker = '.')

#Hard Marg
NoiLin_beta_H, NoiLin_beta0_H = HardMarg(NoiLinear_X,NoiLinear_Y)
NoiLin_beta_xlst_H, NoiLin_beta_ylst_H =  linegen(-3,3,NoiLin_beta_H,NoiLin_beta0_H)
plt.plot(NoiLin_beta_xlst_H,NoiLin_beta_ylst_H, c='g')

lst = classify(NoiLinear_X, NoiLin_beta_H, NoiLin_beta0_H)

NoiLin_beta_S, NoiLin_beta0_S = SoftMarg(NoiLinear_X,NoiLinear_Y,0.5)
NoiLin_beta_xlst_S, NoiLin_beta_ylst_S =  linegen(-3,3, NoiLin_beta_S,NoiLin_beta0_S)
plt.plot(NoiLin_beta_xlst_S,NoiLin_beta_ylst_S, c='y')


lst_NLinH = classify(NoiLinear_X, NoiLin_beta_H, NoiLin_beta0_H) != NoiLinear_Y.T
Train_err_NLinH = np.sum(lst_NLinH,axis=1)/np.shape(NoiLinear_Y)[0]
lst_NLinH_Test = classify(NoiLinear_Test_X, NoiLin_beta_H, NoiLin_beta0_H) != NoiLinear_Test_Y.T
Train_err_NLinH_Test = np.sum(lst_NLinH_Test,axis=1)/np.shape(NoiLinear_Test_Y)[0]

print("Noisy Linear data")
print("Traing error Hard margin: {}".format(Train_err_NLinH))
print("Testing error Hard margin: {}".format(Train_err_NLinH_Test))

lst_NLinS = classify(NoiLinear_X, NoiLin_beta_S, NoiLin_beta0_S) != NoiLinear_Y.T
Train_err_NLinS = np.sum(lst_NLinS,axis=1)/np.shape(NoiLinear_Y)[0]
lst_NLinS_Test = classify(NoiLinear_Test_X, NoiLin_beta_S, NoiLin_beta0_S) != NoiLinear_Test_Y.T
Train_err_NLinS_Test = np.sum(lst_NLinS_Test,axis=1)/np.shape(NoiLinear_Test_Y)[0]
print("Traing error Soft margin: {}".format(Train_err_NLinS))
print("Testing error Hard margin: {}".format(Train_err_NLinS_Test))


# In[23]:


#Noisy Linear
#Red: Training points with y = 1
#Blue: Trainging points with y = -1
#Magenta : Testing points with y = 1
#Black : Testing points with y = -1
#Green: Hard Margin
#Yellow: Soft Margin
Quad_X_p, Quad_X_n = divlst(Quad_X,Quad_Y)
for x in Quad_X_p:
    plt.plot(x[0],x[1], c='r', marker = '.')
for x in Quad_X_n:
    plt.plot(x[0],x[1], c='b', marker = '.')

Quad_X_test_p, Quad_X_test_n = divlst(Quad_Test_X,Quad_Test_Y)
for x in Quad_X_test_p:
    plt.plot(x[0],x[1], c='m', marker = '.')
for x in Quad_X_test_n:
    plt.plot(x[0],x[1], c='k', marker = '.')


#Hard Marg
Quad_beta_H, Quad_beta0_H = HardMarg(Quad_X,Quad_Y)
Quad_beta_xlst_H, Quad_beta_ylst_H =  linegen(0,1,Quad_beta_H,Quad_beta0_H)
plt.plot(Quad_beta_xlst_H,Quad_beta_ylst_H, c='g')

lst = classify(Quad_X, Quad_beta_H, Quad_beta0_H)

Quad_beta_S, Quad_beta0_S = SoftMarg(Quad_X,Quad_Y,0.5)
Quad_beta_xlst_S, Quad_beta_ylst_S =  linegen(0,1, Quad_beta_S,Quad_beta0_S)
plt.plot(Quad_beta_xlst_S,Quad_beta_ylst_S, c='y')

lst_QuadH = classify(Quad_X, Quad_beta_H, Quad_beta0_H) != Quad_Y.T
Train_err_QuadH = np.sum(lst_QuadH,axis=1)/np.shape(Quad_Y)[0]
lst_QuadH_Test = classify(Quad_Test_X, Quad_beta_H, Quad_beta0_H) != Quad_Test_Y.T
Train_err_QuadH_Test = np.sum(lst_QuadH_Test,axis=1)/np.shape(Quad_Test_Y)[0]

print("Quadratic data")
print("Traing error Hard margin: {}".format(Train_err_QuadH))
print("Testing error Hard margin: {}".format(Train_err_QuadH_Test))

lst_QuadS = classify(Quad_X, Quad_beta_S, Quad_beta0_S) != Quad_Y.T
Train_err_QuadS = np.sum(lst_QuadS,axis=1)/np.shape(Quad_Y)[0]
lst_QuadS_Test = classify(Quad_Test_X, Quad_beta_S, Quad_beta0_S) != Quad_Test_Y.T
Train_err_QuadS_Test = np.sum(lst_QuadS_Test,axis=1)/np.shape(Quad_Test_Y)[0]

print("Traing error Soft margin: {}".format(Train_err_QuadS))
print("Testing error Soft margin: {}".format(Train_err_QuadS_Test))


# In[ ]:




