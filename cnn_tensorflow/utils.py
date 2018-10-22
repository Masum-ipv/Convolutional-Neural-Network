from scipy import misc
import tensorflow as tf
import matplotlib.pyplot as plt


def showImage(path):
    img = misc.imread(PATH)
    print("Image Shape: ", img.shape)
    img_tf = tf.Variable(img)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    im = sess.run(img_tf)
    fig = plt.figure()
    plt.imshow(im)
    plt.show()
    
