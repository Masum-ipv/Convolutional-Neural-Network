from __future__ import division
import math, random, os
from skimage import data, io, color
from grapical_view import draw_output
from convolution import connvolution_process
from collections import Counter
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from helper import initialize_parameters_deep, L_model_forward, compute_cost, L_model_backward, update_parameters, predict

inputs = [] # Train data
targets = [] # Train result
test = []
result = [[1 if i == j else 0 for i in range(10)]
                    for j in range(10)]

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009

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

directory = os.fsencode("data/")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    img = io.imread("data/" + filename)
    print("filename ", filename, img.shape)
    if filename.startswith("B0"): 
        serial = 0
    elif filename.startswith("B1"):
        serial = 1
    elif filename.startswith("B2"):
        serial = 2
    elif filename.startswith("B3"):
        serial = 3
    elif filename.startswith("B4"):
        serial = 4
    elif filename.startswith("B5"):
        serial = 5
    elif filename.startswith("B6"):
        serial = 6
    elif filename.startswith("B7"):
        serial = 7
    elif filename.startswith("B8"):
        serial = 8
    elif filename.startswith("B9"):
        serial = 9
    fc = connvolution_process(img)  # Return Fully connected layer
    
    # Append corresponding train data and result
    inputs.append(fc.tolist())
    
    targets.append(result[serial])

inputs = np.array(inputs).T
targets = np.array(targets).T

# Standardize data to have feature values between 0 and 1.
inputs = inputs/255.

print("Inputs Shape : ", inputs.shape)
print("Targets Shape : ", targets.shape)


### CONSTANTS ###
layers_dims = [len(inputs), 30, 20, 10] #  4-layer model
print("layers_dims ", layers_dims)

parameters = L_layer_model(inputs, targets, layers_dims, num_iterations = 2500, print_cost = True)

directory = os.fsencode("test/")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    img = io.imread("test/" + filename)
    print("filename ", filename, img.shape)
    fc = connvolution_process(img)
    test.append(fc.tolist())
#print("\n\n\nBefore", test)

test = np.array(test).T
print("after", test.shape)

test = test/255.
#print(test)
predict(test, parameters)