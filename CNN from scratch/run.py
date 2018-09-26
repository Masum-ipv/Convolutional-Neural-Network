import math, random, os
from skimage import data, io, color
from grapical_view import show
from neural_network import backpropagate, predict
from convolution import connvolution_process

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
    targets.append(result[serial])
    
# print(inputs)
# print(targets)

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


'''---------------------- Backpropagation ---------------------------'''
# 10000 iterations seems enough to converge // 10000
for __ in range(100):
    for input_vector, target_vector in zip(inputs, targets):
        backpropagate(network, input_vector, target_vector)

'''---------------------- Predict the output ---------------------------'''
print(predict(network,fc))
