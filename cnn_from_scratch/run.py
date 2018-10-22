from __future__ import division
import math, random, os
from skimage import data, io, color
from grapical_view import draw_output
from convolution import connvolution_process
from collections import Counter
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from helper import initialize_parameters_deep, L_model_forward, compute_cost, L_model_backward, update_parameters, predict, load_data, load_test_data
from grad_checking import gradient_check_n

data_path = "train_data/"
test_path = "test_data/"
test = []



def L_layer_model(X, Y, layers_dims, learning_rate = 0.06, num_iterations = 5000, print_cost=False):#lr was 0.009

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
        
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)  

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

inputs, targets = load_data("dev_conv.txt", "train_result.txt", data_path)
#temp = np.reshape(inputs[0], (7,7,1))
#print (temp)
#plt.imshow(temp[:, :, 0]).set_cmap("gray")
#plt.show()


inputs = np.array(inputs).T
targets = np.array(targets).T

# Standardize data to have feature values between 0 and 1.
inputs = inputs/255.

print("Inputs Shape : ", inputs.shape)
print("Targets Shape : ", targets.shape)


### CONSTANTS ###
layers_dims = [len(inputs), 30, 20, 10] #  4-layer model
print("layers_dims: ", layers_dims)

parameters = L_layer_model(inputs, targets, layers_dims, num_iterations = 2500, print_cost = True)

test_data, test_file = load_test_data("test_conv.txt", test_path)

test_data = np.array(test_data).T
print("\n\nTest data shape: ", test_data.shape)

test_data = test_data/255.
predict = predict(test_data, parameters)
print("Predict shape: ", predict.shape)

for i in range(0, len(test_file)):
    print (test_file[i], max(predict[i]), np.where(predict[i] == predict[i].max()))