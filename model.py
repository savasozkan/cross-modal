
import tensorflow as tf
from opt import *

def center_norm(x, name='center_norm', is_training=True, decay=0.999, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        
        beta = tf.get_variable('beta', [x.get_shape()[-1]], tf.float32, 
                                initializer=tf.constant_initializer(0.0))
        
        scale = tf.get_variable('scale', [x.get_shape()[-1]], tf.float32, 
                                initializer=tf.constant_initializer(1.0))
        
        batch_mean, batch_var = tf.nn.moments(x, [1, 2], keep_dims=True)
    
    return scale*tf.div(tf.subtract(x, batch_mean), tf.sqrt(tf.add(batch_var, 1e-9))) + beta
    
def network_pre(x, nchannel=2, reuse=False):
    with tf.variable_scope("network") as scope:
        if reuse == True:
            scope.reuse_variables()

        sz = x.get_shape().as_list() 
                      
        e0 = conv2d(x, 32, name='l_eh0_conv') ## w/2
        e0 = batch_norm(e0, name='l_eh0_bn', training=btrain)
        e0 = prelu(e0, name='l_eh0_prelu')

        e1 = conv2d(e0, 64, name='l_eh1_conv') ## w/4
        e1 = batch_norm(e1, name='l_eh1_bn', training=btrain)
        e1 = prelu(e1, name='l_eh1_prelu')
    
        e2 = conv2d(e1, 128, name='l_eh2_conv') ## w/8
        e2 = batch_norm(e2, name='l_eh2_bn', training=btrain)
        e2 = prelu(e2, name='l_eh2_prelu')
        
        e3 = conv2d(e2, 256, name='l_eh3_conv') ## w/16
        e3 = batch_norm(e3, name='l_eh3_bn', training=btrain)
        e3 = prelu(e3, name='l_eh3_prelu')

        e4 = conv2d(e3, 256, stride=1, name='l_eh4_conv') ## w/16
        e4 = batch_norm(e4, name='l_eh4_bn', training=btrain)
        e4 = prelu(e4, name='l_eh4_prelu')
                
        h1 = tf.image.resize_images(e4, [sz[1]/8, sz[2]/8])
        h1 = conv2d(h1, 128, stride=1, name='l_dh0_conv') + e2
        h1 = batch_norm(h1, name='l_dh0_bn', training=btrain)
        h1 = tf.nn.relu(h1) 

        h2 = tf.image.resize_images(h1, [sz[1]/4, sz[2]/4])
        h2 = conv2d(h2, 64, stride=1, name='l_dh1_conv') + e1
        h2 = batch_norm(h2, name='l_dh1_bn', training=btrain)
        h2 = tf.nn.relu(h2) 
        
        h3 = tf.image.resize_images(h2, [sz[1]/2, sz[2]/2])
        h3 = conv2d(h3, 32, stride=1, name='l_dh2_conv') + e0
        h3 = batch_norm(h3, name='l_dh2_bn', training=btrain)
        h3 = tf.nn.relu(h3) 
               
        h4 = tf.image.resize_images(h3, [sz[1], sz[2]])
        h4 = conv2d(h4, nchannel, stride=1, name='l_dh3_conv')
        
    return h4

def network_post(x, nchannel=2, reuse=False):
    with tf.variable_scope("network") as scope:
        if reuse == True:
            scope.reuse_variables()

        sz = x.get_shape().as_list() 
                
        e0 = center_norm(x, 'l_eh0_center')
        
        e0 = conv2d(e0, 32, kernel=11, name='l_eh0_conv') ## w/2
        e0 = prelu(e0, name='l_eh0_prelu')

        e1 = batch_norm(e0, name='l_eh1_bn', training=btrain)
        e1 = conv2d(e1, 64, name='l_eh1_conv') ## w/4
        e1 = prelu(e1, name='l_eh1_prelu')
    
        e2 = batch_norm(e1, name='l_eh2_bn', training=btrain)
        e2 = conv2d(e2, 128, name='l_eh2_conv') ## w/8
        e2 = prelu(e2, name='l_eh2_prelu')
        
        e3 = batch_norm(e2, name='l_eh3_bn', training=btrain)
        e3 = conv2d(e3, 256, name='l_eh3_conv') ## w/16
        e3 = prelu(e3, name='l_eh3_prelu')

        e4 = batch_norm(e3, name='l_eh4_bn', training=btrain)
        e4 = conv2d(e4, 256, stride=1, name='l_eh4_conv') ## w/16
        e4 = prelu(e4, name='l_eh4_prelu')
                
        h1 = tf.image.resize_images(e4, [sz[1]/8, sz[2]/8])
        h1 = batch_norm(h1, name='l_dh0_bn', training=btrain)
        h1 = conv2d(h1, 128, stride=1, name='l_dh0_conv') + e2
        h1 = prelu(h1, name='l_dh0_prelu') 

        h2 = tf.image.resize_images(h1, [sz[1]/4, sz[2]/4])
        h2 = batch_norm(h2, name='l_dh1_bn', training=btrain)
        h2 = conv2d(h2, 64, stride=1, name='l_dh1_conv') + e1
        h2 = prelu(h2, name='l_dh1_prelu') 
        
        h3 = tf.image.resize_images(h2, [sz[1]/2, sz[2]/2])
        h3 = batch_norm(h3, name='l_dh2_bn', training=btrain)
        h3 = conv2d(h3, 32, stride=1, name='l_dh2_conv') + e0
        h3 = prelu(h3, name='l_dh2_prelu') 
               
        h4 = tf.image.resize_images(h3, [sz[1], sz[2]])
        h4 = conv2d(h4, nchannel, stride=1, name='l_dh3_conv')
        
    return h4

def discriminator_encoder(x, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        
        h0 = conv2d(x, 32, name='d_eh0_conv')
        h0 = batch_norm(h0, name='d_eh0_bn')
        h0 = prelu(h0, name='d_eh0_prelu')

        h1 = conv2d(h0, 32, name='d_eh1_conv')
        h1 = batch_norm(h1, name='d_eh1_bn')
        h1 = prelu(h1, name='d_eh1_prelu')
    
        h2 = conv2d(h1, 32, name='d_eh2_conv')
        h2 = batch_norm(h2, name='d_eh2_bn')
        h2 = prelu(h2, name='d_eh2_prelu')
        
        h3 = conv2d(h2, 32, name='d_eh3_conv')
        h3 = batch_norm(h3, name='d_eh3_bn')
        h3 = prelu(h3, name='d_eh3_prelu')

        h4 = conv2d(h3, 32, name='d_eh4_conv')
        h4 = batch_norm(h4, name='d_eh4_bn')
        h4 = prelu(h4, name='d_eh4_prelu')
        
        sz  = h4.get_shape().as_list()                
        hf1 = tf.reshape(h4, [sz[0], sz[1]*sz[2]*sz[3]])
        hf1 = linear(hf1, 256, name='d_ehf1_lin')
        hf1 = prelu(hf1, name='d_ehf1_prelu')
                
        hf = linear(hf1, 1, name='d_hf_lin')
        
    return hf

def loss_dcgan(x, gen):
    real_d = discriminator_encoder(x,   reuse=False)
    fake_d = discriminator_encoder(gen, reuse=True)
   
    d_r_cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=real_d, labels=tf.ones_like(real_d)) )
    d_f_cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_d, labels=tf.zeros_like(fake_d)) )
    
    g_f_cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_d,  labels=tf.ones_like(fake_d)) )
    
    d_cost = d_r_cost + d_f_cost
    g_cost = g_f_cost 

    return g_cost, d_cost
