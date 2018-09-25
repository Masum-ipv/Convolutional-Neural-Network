from skimage import data, io, color
import numpy
from utils import conv, relu, pooling
from grapical_view import show

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
# print(l1_filter.shape)



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

'''---------------------- Graphing results ---------------------------'''
show(img, l1_feature_map, l1_feature_map_relu, l1_feature_map_relu_pool, l2_feature_map, l2_feature_map_relu, l2_feature_map_relu_pool, l3_feature_map, l3_feature_map_relu, l3_feature_map_relu_pool)
