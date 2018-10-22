from scipy import misc
import tensorflow as tf
import matplotlib.pyplot as plt


def showImage(path):
    img = misc.imread('01.png')
    print img.shape
    img_tf = tf.Variable(img)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    im = sess.run(img_tf)
    fig = plt.figure()
    plt.imshow(im)
    plt.show()
    
