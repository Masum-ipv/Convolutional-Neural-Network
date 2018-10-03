from __future__ import division
import math, random, os
from skimage import data, io, color
from grapical_view import draw_output
from convolution import connvolution_process
from collections import Counter
from functools import partial
import matplotlib
import numpy as np
from helper import initialize_parameters_deep, L_model_forward, compute_cost, L_model_backward, update_parameters
    
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
    
inputs = [] # Train data
targets = [] # Train result
result = [[1 if i == j else 0 for i in range(10)]
                    for j in range(10)]

directory = os.fsencode("data/")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    img = io.imread("data/" + filename)
    print("filename ", filename)
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
    inputs.append(fc)
    print("Inputs fc : ", fc)
    targets.append(result[serial])

inputs = np.array(inputs).T
targets = np.array(targets).T

# Standardize data to have feature values between 0 and 1.
#inputs = inputs/255.

#print("Inputs Shape : ", inputs)
#print("Targets Shape : ", targets)


### CONSTANTS ###
layers_dims = [inputs.shape[0], 30, 20, 10] #  4-layer model

parameters = L_layer_model(inputs, targets, layers_dims, num_iterations = 25, print_cost = True)

































'''
random.seed(0)   # to get repeatable results
input_size = len(fc)  # each input is a vector of length 49
num_hidden = 5   # we'll have 5 neurons in the hidden layer
output_size = 10 # we need 10 outputs for each input

# each hidden neuron has one weight per input, plus a bias weight
hidden_layer = [[random.random() for __ in range(input_size + 1)]
                for __ in range(num_hidden)]

# each output neuron has one weight per hidden neuron, plus a bias weight
output_layer = [[random.random() for __ in range(num_hidden + 1)]
                for __ in range(output_size)]

# the network starts out with random weights
network = [hidden_layer, output_layer]

"""---------------------- Backpropagation ---------------------------"""
# 10000 iterations seems enough to converge // 10000
for __ in range(10):
    for input_vector, target_vector in zip(inputs, targets):
        backpropagate(network, input_vector, target_vector)
"""---------------------- Predict the output ---------------------------"""

directory = os.fsencode("test/")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    img = io.imread("test/" + filename)
    fc = connvolution_process(img)  # Return Fully connected layer
    print(fc)
    with open('file.txt', 'w') as f:
        for item in fc:
            f.write("%s\n" % item)
    pred_list = predict(network, fc)
    print("\n\n\nPrediction : ", pred_list)
    print("*************** See the Result Folder to see Test image result with image ************************")  
    draw_output(img, "Output_" + filename, max(pred_list), pred_list.index(max(pred_list)))

'''