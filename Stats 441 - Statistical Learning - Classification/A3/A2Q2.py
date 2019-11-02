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

#Question 2
Q2_train_Ion_x = loadmat('Ion.trin.mat')['Xtrain']
Q2_train_Ion_y = loadmat('Ion.trin.mat')['ytrain']
Q2_test_Ion_x = loadmat('Ion.test.mat')['Xtest']
Q2_test_Ion_y = loadmat('Ion.test.mat')['ytest']


# Dataset: 176 element
# Want to apply PCA, but can't


#a)

#Forward feeding 
# input: the x vector of the input
# Layer: List of the weight associated with each layer
# layfun: List of the propergation function
# Output: The result of the back prop
def Q2_forwardfeed(input,Layer,layfun,layer_size):
    for x in range(0, layer_size):
        input = layfun[x](np.matmul(input,Layer[x]))
    return input

# The function takes in the input layer, the y vecotor,
# The activation function and the derivative of the activation function
# And as well as the weight for each layer, and output the derivative of the L_2 
# norm with respect to the weight

# Output: 
# delta_lst: List of the matrix of the change in u_{il}
def Q2_backprop(input ,y ,Layers ,layfun ,layfunp,Layer_size):

    #Applying forward propagation 
    a_i = []
    z_i = [input]
    Layer = np.copy(Layers)

    
    for x in range(0, Layer_size):
        a_i.append(np.matmul(input,Layer[x]))
        input  = layfun[x](a_i[x])
        z_i.append(input)

        
    #Setting up for backprop
    delta_lst = [np.multiply(-2,(y-z_i[Layer_size]))]
    
    #Calculating delta 
    for x in range(0,Layer_size):
        sigma_p_a_i = layfunp[Layer_size-1-x](a_i[Layer_size-1-x])
        sigma_delta_u = np.matmul(Layer[Layer_size-1-x],np.transpose(delta_lst[0]))
        delta_lst.insert(0,np.matmul(np.transpose(sigma_p_a_i),np.transpose(sigma_delta_u)))


    #delta
    u_new_lst = []
    
    for x in range(0, Layer_size):
        det = np.matmul(np.transpose(z_i[x]),delta_lst[x+1])
        u_new_lst.append(det)
    return u_new_lst

def Q2_train_b(input,y,Layer,layfun, layfunp,itera,decay_factor, layer_size):

    Training_err = []
    Test_err = []
    sum_err = []
    iterat = []


    for r in range(0, itera):
        #initi the layers
        delta_lst = []
        for x in range(0, layer_size):
            delta_lst.append(np.zeros(np.shape(Layer[x])))
        
        #Calculate the backprop for each of the given point
        for x in range(0,np.shape(input)[0]):
            delta_u =  Q2_backprop(input[x:x+1,:] ,y[x] ,Layer ,layfun ,layfunp,layer_size)
            for z in range(0, layer_size ):
                delta_lst[z] = np.add(np.multiply(1/np.shape(input)[0] , delta_u[z]),delta_lst[z])

        #Updating the weight
        for z in range(0, layer_size):
            Sum_weight = np.divide(Layer[z],np.sqrt(np.sum(np.multiply(Layer[z],Layer[z] ), axis = 0)))
            Layer[z] = np.add(np.subtract(Layer[z], delta_lst[z]),np.multiply(decay_factor ,Sum_weight))

        yp_train = Q2_forwardfeed(input, Layer,Q2_Act_func,2)
        Q2_train_result =  np.transpose(yp_train)>np.full((176,),0.5)
        Q2_train_error = np.transpose(Q2_train_result)!=Q2_train_Ion_y

       # print("Training error: {}".format(sum(Q2_train_error)/176))

        yp_test = Q2_forwardfeed(np.transpose(Q2_test_Ion_x), Layer,Q2_Act_func,2)
        Q2_test_result = np.transpose(yp_test)>np.full((175,),0.5)
        Q2_test_error = np.transpose(Q2_test_result)!=Q2_test_Ion_y

        #print("Testing error: {}".format(sum(Q2_test_error)/175))
        
        iterat.append(r)
        Training_err.append(sum(Q2_train_error)/176)
        Test_err.append(sum(Q2_test_error)/175)
        sum_err.append(sum(Q2_train_error)/176 - sum(Q2_test_error)/175)


    plt.plot(iterat,Training_err, 'r', label='Training error')
    plt.plot(iterat,Test_err, 'b', label='Testing error')
    plt.plot(iterat,sum_err, 'y', label='Training + Testing error error')
    plt.draw()
    plt.show()
    plt.clf()

    return Layer

#Change it to e
def Q2_train(input,y,Layer,layfun, layfunp,itera,decay_factor, layer_size ):

    for r in range(0, itera):
        # initi the layers
        delta_lst = []
        for x in range(0, layer_size):
            delta_lst.append(np.zeros(np.shape(Layer[x])))

        # Calculate the backprop for each of the given point
        for x in range(0,np.shape(input)[0]):
            delta_u =  Q2_backprop(input[x:x+1,:] ,y[x] ,Layer ,layfun ,layfunp,layer_size)
            for z in range(0, layer_size ):
                delta_lst[z] = np.add(np.multiply(1/np.shape(input)[0] , delta_u[z]),delta_lst[z])

        # Updating the weight
        for z in range(0, layer_size):
            #Normalize the Weight
            Sum_weight = np.divide(Layer[z],np.sqrt(np.sum(np.multiply(Layer[z],Layer[z] ), axis = 0)))
            Layer[z] = np.add(np.subtract(Layer[z], delta_lst[z]),np.multiply(decay_factor ,Sum_weight))

        '''
        yp = Q2_forwardfeed(input, Layer,Q2_Act_func,2)
        Q2_train_result =  np.transpose(yp)>np.full((176,),0.5)
        Q2_train_error =np.transpose(Q2_train_result)!=Q2_train_Ion_y


        print("Training error: {}".format(sum(Q2_train_error)))
        '''

    return Layer

#b)
Q2_Act_func = [(lambda y: list(map((lambda x: list(map(Q2_relu,x))), y)))
                ,(lambda y: list(map((lambda x: list(map(Q2_sigmoid,x))), y)))]

Q2_Act_funcp = [(lambda y: list(map((lambda x: list(map(Q2_relu_p,x))), y)))
                ,(lambda y: list(map((lambda x: list(map(Q2_relu_p,x))), y)))]



def Q2_sigmoid(X):
    return 1/(1 + np.e**(-X))

def Q2_sigmoid_p(X):
    return np.multiply(Q2_sigmoid(X), (1- Q2_sigmoid(X)))

def Q2_relu(X):
    if(X < 0):
        return X
    return 0

def Q2_relu_p(X):
    if(X < 0):
        return 1
    return 0


def Q2b():
    
    # Initialize a N network
    np.random.seed(123)
    Q2_Layer = [np.random.normal(size = (33,16)) , np.random.normal(size = (16,1))]

    # Have different alpha
    Q2_train_b(np.transpose(Q2_train_Ion_x),Q2_train_Ion_y,Q2_Layer,Q2_Act_func, Q2_Act_funcp ,200,0, 2 )
    Q2_train_b(np.transpose(Q2_train_Ion_x),Q2_train_Ion_y,Q2_Layer,Q2_Act_func, Q2_Act_funcp ,200,0.1, 2 )
    Q2_train_b(np.transpose(Q2_train_Ion_x),Q2_train_Ion_y,Q2_Layer,Q2_Act_func, Q2_Act_funcp ,200,0.5, 2 )
    Q2_train_b(np.transpose(Q2_train_Ion_x),Q2_train_Ion_y,Q2_Layer,Q2_Act_func, Q2_Act_funcp ,200,1, 2 )



def Q2c():

    #Empty list for collection
    Q2_train_error_lst = []
    Q2_test_error_lst = []
    Q2_error_sum_lst = []
    Q2_Layer_lst = []

    for x in range(1, 30):
        #Hidden Layersize
        Q2_HLsize = x

        # Initialize a N network
        np.random.seed(123)
        Q2_Layer = [np.random.normal(size = (33,Q2_HLsize)),np.random.normal(size = (Q2_HLsize,1))]

        #Some issue with calling, use copy matrix instead
        Q2_train_Dat_x = np.copy(Q2_train_Ion_x)
        Q2_train_Dat_y = np.copy(Q2_train_Ion_y)
        Q2_test_Dat_x = np.copy(Q2_test_Ion_x)
        Q2_test_Dat_y = np.copy(Q2_test_Ion_y)

        #Training data
        Q2_Layer = Q2_train(np.transpose(Q2_train_Dat_x), Q2_train_Dat_y, Q2_Layer, Q2_Act_func, Q2_Act_funcp, 100,0,2)

        #Calculating training error
        Q2_train_net = Q2_forwardfeed(np.transpose(Q2_train_Dat_x), Q2_Layer,Q2_Act_func,2)

        Q2_train_result=  np.transpose(Q2_train_net)>np.full((176,),0.5)
        Q2_train_error =np.transpose(Q2_train_result)!=Q2_train_Dat_y

        
        print("n = {}".format(x))
        print("Training error: {}".format(sum(Q2_train_error)/176))

        Q2_test = Q2_forwardfeed(np.transpose(Q2_test_Dat_x), Q2_Layer,Q2_Act_func,2)

        Q2_test_result =  np.transpose(Q2_test)>np.full((175,),0.5)
        Q2_test_error =np.transpose(Q2_test_result)!=Q2_test_Dat_y

        print("Test error: {}".format(sum(Q2_test_error)/175))
        print("Total: {}".format(sum(Q2_test_error)/175 + sum(Q2_train_error)/176))

        '''
        print("\hline")
        print("{} & {} & {} & {} \\\\ ".format(x, sum(Q2_train_error)/176, sum(Q2_test_error)/175,sum(Q2_train_error)/176 + sum(Q2_test_error)/175 ))
        '''
        Q2_train_error_lst.append(sum(Q2_train_error)/176)
        Q2_test_error_lst.append(sum(Q2_test_error)/175)
        Q2_error_sum_lst.append( sum(Q2_test_error)/175 + sum(Q2_train_error)/176)
        Q2_Layer_lst.append(x)

    plt.plot(Q2_Layer_lst,Q2_train_error_lst,'r', label='Training error')
    plt.plot(Q2_Layer_lst,Q2_test_error_lst,'b', label='Testing error')
    plt.plot(Q2_Layer_lst, Q2_error_sum_lst,'g', label='Testing + Training error error')
    plt.draw()
    plt.show()
    plt.clf()

#Q2b()

#Q2d

Q2_HLsize = 23

# Initialize a N network
np.random.seed(123)
Q2_Layer = [np.random.normal(size = (33,Q2_HLsize)),np.random.normal(size = (Q2_HLsize,1))]

#Training data
Q2_Layer= Q2_train(np.transpose(Q2_train_Ion_x),Q2_train_Ion_y,Q2_Layer,Q2_Act_func, Q2_Act_funcp ,120,0.1, 2 )

Q2_train_net = Q2_forwardfeed(np.transpose(Q2_test_Ion_x), Q2_Layer,Q2_Act_func,2)

#first digit is the prediction, second is actual
y_00 = 0
y_01 = 0
y_10 = 0
y_11 = 0

for x in range(1, 175):
    if(Q2_train_net[x][0] <= 0.5):
        if(Q2_test_Ion_y[x] == 0):
            y_00 += 1
        else:
            y_01 += 1
    else:
        if(Q2_test_Ion_y[x] == 1):
            y_11 += 1
        else:
            y_10 += 1

print("{} & {} & {} & {}".format(y_00,y_01,y_10,y_11))

print(sum(Q2_test_Ion_y))