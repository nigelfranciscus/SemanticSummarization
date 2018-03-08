import tensorflow as tf
import numpy

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))