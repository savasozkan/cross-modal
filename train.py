
import tensorflow as tf
from opt import *
from model import *

learning_rate  = 0.001
training_iters = 100000
batch_size     = 6
display_step   = 100

dim            = 256
depth          = 1

xi = tf.placeholder("float", [batch_size, dim, dim, 2*depth+1])
xo = tf.placeholder("float", [batch_size, dim, dim, 2])

#####

xop = network_post(xi)
xops = tf.nn.softmax(xop)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=xop, labels=xo))

xr = tf.concat([xc, xo], axis=3)
xf = tf.concat([xc, xops], axis=3)

g_cost, d_cost = loss_dcgan(xr, xf)

vars_list = tf.trainable_variables()
gen_vars  = [var for var in vars_list if 'l_' in var.name]
disc_vars = [var for var in vars_list if 'd_' in var.name]

optimizerc = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss+0.001*g_cost, var_list=gen_vars)
optimizerd = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(d_cost, var_list=disc_vars)

#####
