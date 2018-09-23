from skimage import data, io, color
import numpy
from matplotlib import pyplot as plt
from utils import conv

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



'''---------------------- Convolutional Layer ---------------------------'''
l1_feature_map = conv(img, l1_filter)
print(l1_feature_map)