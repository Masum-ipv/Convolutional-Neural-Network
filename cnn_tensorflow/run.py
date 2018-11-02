import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir
from os.path import isfile,join
import cv2
import math
from scipy import misc, ndimage


#Data Preparation
path = "Convolutional-Neural-Network/cnn_tensorflow/"

def model(X_train, Y_train, X_test, Y_test, learning_rate, num_epochs, minibatch_size, print_cost = True):

    tf.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    X, Y = create_placeholders(n_x, n_y)
    
    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)
    
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                _ , minibatch_cost = sess.run([optimizer, cost], 
                                             feed_dict={X: minibatch_X, 
                                                        Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters



#Data Preparation
TrainFolderNames = []
TestFolderNames = []
trainData = []
trainResponseData = []
testData=[]
testResponseData=[]
for i in range(10):
    TrainFolderNames.append(path+ "train_data/" + str(i))
    TestFolderNames.append(path+ "test_data/" + str(i))


k=0
for folder in TrainFolderNames:
    for image in listdir(folder):
        img =cv2.imread(join(folder,image))
        img = cv2.resize(img,(32,32))
        trainData.append(img.flatten())
        trainResponseData.append(k)
    k=k+1
    
k=0
for folder in TestFolderNames:
    for image in listdir(folder):
        img =cv2.imread(join(folder,image))
        img = cv2.resize(img,(32,32))
        testData.append(img.flatten())
        testResponseData.append(k)
    k=k+1

X_train = np.float32(trainData).T # (2183, 3072)
X_test = np.float32(testData).T

# OneHot Encoder
n_values = np.max(trainResponseData) + 1
Y_train = np.eye(n_values)[trainResponseData].T
Y_test = np.eye(n_values)[testResponseData].T

# Normalize image vectors
X_train = X_train/255.
X_test = X_test/255.

print("X_train shape ", X_train.shape)
print("Y_train shape ", Y_train.shape)

print("X_test shape ", X_test.shape)
print("Y_test shape ", Y_test.shape)


parameters = model(X_train, Y_train, X_test, Y_test, 0.0001, 600, 32)




# Predict all test image
X_test = X_test.T
for i in range (30):
  my_image = X_test[i].reshape(3072, 1)
  my_image_prediction = predict(my_image, parameters)
  print("Orginal digit: " + str(testResponseData[i]))
  #plt.imshow()
  print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)) + "\n")


# Predict my own image
fname = 'Convolutional-Neural-Network/cnn_tensorflow/test_data/0/0.tif'
image = cv2.imread(fname)
print(image.shape)
image = cv2.resize(image,(64,64))
my_image = cv2.resize(image,(32,32))
print(my_image.shape)
my_image = my_image.reshape((1, 32*32*3)).T
my_image = my_image/255.
#my_image = misc.imresize(image, size=(32,32)).reshape((1, 32*32*3)).T   # 32
print("my_image predict shape ",my_image.shape)
my_image_prediction = predict(my_image, parameters)
print(my_image_prediction)
plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
