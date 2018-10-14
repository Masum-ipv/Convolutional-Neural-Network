## Implementation of Bangla Handwriting Recognition From Scratch

Store convolutional layer output in text file (As convolutional layer time comsuming)

### The major steps involved are as follows:

1. Reading the input image.

2. Preparing filters.

3. Conv layer: Convolving each filter with the input image.

4. ReLU layer: Applying ReLU activation function on the feature maps (output of conv layer).

5. Max Pooling layer: Applying the pooling operation on the output of ReLU layer.

6. Stacking conv, ReLU, and max pooling layers.

7. Convert convolution layer to Fully connected layer

8. Backpropagation Fully connected layer

9. Predict the output




#### Run
```sh
$ python run.py
```
##### Accuracy 80%

(!plot)()

* data Folder: Train Data
* test Folder: Test Data


#### Resources:
* [CNN] - Implementaition of CNN without using any library with tutorial
* [Source Code] - Implementation of CNN without using any library github link
 

   [CNN]: <https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a>
   [Source Code]: <https://github.com/zishansami102/CNN-from-Scratch>
