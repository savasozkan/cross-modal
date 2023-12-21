
import tensorflow as tf
from scipy.ndimage import measurements as measure
import numpy as np
from tensorflow.contrib.learn.python.learn import trainable

def dice_coeff(labels, predictions, weights=None, name="diceCoeff"):
    #with tf.variable_scope(name):
    TP = tf.metrics.true_positives(labels=labels,predictions=predictions,weights=weights)
    FN = tf.metrics.false_negatives(labels=labels,predictions=predictions,weights=weights)
    FP = tf.metrics.false_positives(labels=labels,predictions=predictions,weights=weights)
    dice_result = tf.divide(tf.add(TP,TP),tf.add(tf.add(FP,FN),tf.add(TP,TP)))

    return dice_result

def conv2d(input_, output_dim, kernel=5, stride=2, stddev=0.02, name="conv2d", padding='SAME', isbias=True):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel, kernel, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        
        conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], 
                                 initializer=tf.constant_initializer(0.0), trainable=isbias)
        conv = tf.nn.bias_add(conv, biases)
        
    return conv
    
def deconv2d(input_, output_shape, kernel=5, stride=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel, kernel, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, stride, stride, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    return deconv

def batch_norm(x, epsilon=1e-5, momentum = 0.9, name="batch_norm", training=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=momentum, 
                      updates_collections=None,
                      epsilon=epsilon,
                      scale=True,
                      is_training=training,
                      scope=name)

def instance_norm(x, name='const_norm'):
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, 1e-9)))

def instance_norm_wrapper(x, name='instance_norm', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
            
        scale = tf.get_variable('scale', [x.get_shape()[-1]], tf.float32, 
                                initializer=tf.constant_initializer(1.0))
    
        beta = tf.get_variable('beta', [x.get_shape()[-1]], tf.float32, 
                                initializer=tf.constant_initializer(0.0))

        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        
        with tf.control_dependencies([mean, var]):
            return tf.nn.batch_normalization(x, mean, var, beta, scale, 1e-7)

def cond_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)
    
def dropout(x, keep_prob=0.5, training=True):
    #prob = tf.cond(training, keep_prob, 1.0)
    return tf.nn.dropout(x, keep_prob=keep_prob)

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def relu(x, name='relu'):
    return tf.nn.relu(x)

def prelu(x, name='prelu'):
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', [x.get_shape()[-1]], tf.float32, 
                                initializer=tf.constant_initializer(0.01))
    
    beta = tf.minimum(0.1, tf.maximum(beta, 0.01))
        
    return tf.maximum(x, beta*x)

def maxpool2d(x, kernel=2, stride=2, name='max_pool'):
    return tf.nn.max_pool(x, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding='SAME')

def linear(input_, output_size, name=None, stddev=0.02, bias_start=0.0, isbias=True):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        
        lin = tf.matmul(input_, matrix)
        bias = tf.get_variable("bias", [output_size],
                initializer=tf.constant_initializer(bias_start), trainable=isbias)
        
        lin = tf.nn.bias_add(lin, bias)

    return lin

def subpixel(X, r, n_out_channel):
    if n_out_channel >= 1:
        assert int(X.get_shape()[-1]) == (r ** 2) * n_out_channel, 'Invalid Params'
        bsize, a, b, c = X.get_shape().as_list()
        bsize = tf.shape(X)[0] # Handling Dimension(None) type for undefined batch dim
        Xs=tf.split(X,r,3) #b*h*w*r*r
        Xr=tf.concat(Xs,2) #b*h*(r*w)*r
        X=tf.reshape(Xr,(bsize,r*a,r*b,n_out_channel)) # b*(r*h)*(r*w)*c
    else:
        print('Invalid Dim.')
    return X


def max_region(im):
    im2 = np.copy(im)
    blobs, num_features = measure.label(im2)

    if num_features == 0:
        return im

    nlist = []
    for idx in xrange(num_features):
        s = (blobs == idx + 1)
        nlist.append(np.sum(s))

    pos = np.argmax(nlist)

    for idx in xrange(num_features):
        if pos == idx:
            continue

        s = (blobs == idx + 1)
        im2[s] = 0.

    return im2