from skimage import data, io, color
import numpy
import math, random
from utils import conv, relu, pooling
from grapical_view import show
from neural_network import backpropagate, predict

'''
The major steps involved are as follows:

1. Reading the input image.

2. Preparing filters.

3. Conv layer: Convolving each filter with the input image.

4. ReLU layer: Applying ReLU activation function on the feature maps (output of conv layer).

5. Max Pooling layer: Applying the pooling operation on the output of ReLU layer.

6. Stacking conv, ReLU, and max pooling layers.

'''


'''----------------------Reading the image-------------------------------'''
img = io.imread("data/B0_01.jpg")
# print(img.shape) # 3D image

# Converting the image into gray.
img = color.rgb2gray(img)
# print(img.shape) # 2D image
# io.imshow(img)
# plt.show()



'''----------------------Preparing Filter-------------------------------'''
l1_filter = numpy.zeros((2,3,3))
# Vertical ditector Filter
l1_filter[0, :, :] = numpy.array([[[-1, 0, 1],

                                   [-1, 0, 1],

                                   [-1, 0, 1]]])
# Horizontal ditector Filter
l1_filter[1, :, :] = numpy.array([[[1,   1,  1],

                                   [0,   0,  0],

                                   [-1, -1, -1]]])
# print(l1_filter)

'''---------------------- Convolutional Layer 1 ---------------------------'''
l1_feature_map = conv(img, l1_filter)
print("l1_feature_map", l1_feature_map.shape)
l1_feature_map_relu = relu(l1_feature_map)
print("l1_feature_map_relu", l1_feature_map_relu.shape)
l1_feature_map_relu_pool = pooling(l1_feature_map_relu, 2, 2)
print("l1_feature_map_relu_pool", l1_feature_map_relu_pool.shape)
print("**End of conv layer 1**\n\n")

'''---------------------- Convolutional Layer 2 ---------------------------'''
l2_filter = numpy.random.rand(3, 5, 5, l1_feature_map_relu_pool.shape[-1])
l2_feature_map = conv(l1_feature_map_relu_pool, l2_filter)
print("l2_feature_map", l2_feature_map.shape)
l2_feature_map_relu = relu(l2_feature_map)
print("l2_feature_map_relu", l2_feature_map_relu.shape)
l2_feature_map_relu_pool = pooling(l2_feature_map_relu, 2, 2)
print("l2_feature_map_relu_pool", l2_feature_map_relu_pool.shape)
print("**End of conv layer 2**\n\n")

'''---------------------- Convolutional Layer 3 ---------------------------'''
l3_filter = numpy.random.rand(1, 7, 7, l2_feature_map_relu_pool.shape[-1])
l3_feature_map = conv(l2_feature_map_relu_pool, l3_filter)
print("l3_feature_map", l3_feature_map.shape)
l3_feature_map_relu = relu(l3_feature_map)
print("l3_feature_map_relu", l3_feature_map_relu.shape)
l3_feature_map_relu_pool = pooling(l3_feature_map_relu, 2, 2)
print("l3_feature_map_relu_pool", l3_feature_map_relu_pool.shape)
print("**End of conv layer 3**\n\n")

'''---------------------- Graphing results of convolution ---------------------------'''
# show(img, l1_feature_map, l1_feature_map_relu, l1_feature_map_relu_pool, l2_feature_map, l2_feature_map_relu, l2_feature_map_relu_pool, l3_feature_map, l3_feature_map_relu, l3_feature_map_relu_pool)




'''---------------------- Fully Connected layer ---------------------------'''
print("**Fully connected layer(Convolutional layer to Fully connected layer**")
fc = l3_feature_map_relu_pool.reshape(-1)
print(fc.shape)

inputs = [fc]

targets = [[1 if i == j else 0 for i in range(10)]
               for j in range(10)]
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
# print("network1",network)

# 100 iterations seems enough to converge // 10000
for __ in range(10000):
    for input_vector, target_vector in zip(inputs, targets):
        backpropagate(network, input_vector, target_vector)
# print("network2",network)


print(predict(network,fc))