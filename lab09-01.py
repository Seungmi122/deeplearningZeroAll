import tensorflow as tf 
import numpy as np  

##broadcasting
##helps multiply matrices regardless of their size

matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant(3.)
(matrix1+matrix2).eval()