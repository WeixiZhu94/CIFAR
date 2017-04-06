import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
from taxomony import _cat1, _cat2, _cat3, _cat4, _cat5, _cat1_logits, _cat2_logits, _cat3_logits, _cat4_logits, _cat5_logits, _logsumexp
regularizer = slim.l2_regularizer(0.0002)


def _si_conv(net, channel, prefix):
   net = slim.layers.conv2d(net, channel, [3,3], scope=prefix + 'conv_1', activation_fn=tf.nn.elu, normalizer_fn=slim.layers.batch_norm, weights_regularizer=regularizer)
   return net

def _bi_conv(net, channel, prefix):
   net = slim.layers.conv2d(net, channel, [3,3], scope=prefix + 'conv_1', activation_fn=tf.nn.elu, normalizer_fn=slim.layers.batch_norm, weights_regularizer=regularizer)
   net = slim.layers.conv2d(net, channel, [3,3], scope=prefix + 'conv_2', activation_fn=tf.nn.elu, normalizer_fn=slim.layers.batch_norm, weights_regularizer=regularizer)
   return net

def _residual(net, in_filter, out_filter, prefix):
   # ori_net : not activated; net -> BN -> RELU
   with tf.variable_scope(prefix+'_pre_act'):
      ori_net = net
      net = slim.layers.batch_norm(net)
      net = tf.nn.elu(net)
   with tf.variable_scope(prefix+'_residual'):
      # net -> Weight -> BN -> RELU
      net = slim.layers.conv2d(net, out_filter, [3,3], scope='conv_1',  activation_fn=tf.nn.elu, normalizer_fn=slim.layers.batch_norm, weights_regularizer=regularizer)
      # net -> Weight
      net = slim.layers.conv2d(net, out_filter, [3,3], scope='conv_2', activation_fn=None, weights_regularizer=regularizer)
   with tf.variable_scope(prefix+'_res_add'):
      if in_filter != out_filter:
         ori_net = tf.nn.avg_pool(ori_net, [1,1,1,1], [1,1,1,1], 'VALID')
         ori_net = tf.pad(ori_net, [[0,0],[0,0],[0,0],[(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      net += ori_net
   return net

def network(net, labels):

   net = _si_conv(net, 16, 'res_init')
   
   net = _residual(net, 16, 16, 'unit_16_1')
   for i in range(7):
      net = _residual(net, 16, 16, 'unit_16_%s' % (i+2))
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_1')

   net = _residual(net, 16, 32, 'unit_32_1')
   for j in range(7):
      net = _residual(net, 32, 32, 'unit_32_%s' % (j+2))
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_2')

   net = _residual(net, 32, 64, 'unit_64_1')
   for k in range(7):
      net = _residual(net, 64, 64, 'unit_64_%s' % (k+2))
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_3')

   with tf.variable_scope('res_last'):
      net = slim.layers.batch_norm(net)
      net = tf.nn.elu(net)
      net = tf.reduce_mean(net, [1,2])

   logits = slim.layers.fully_connected(net, 10, activation_fn=None, scope='logits', weights_regularizer=regularizer)

   #calculating loss
   loss = tf.losses.sparse_softmax_cross_entropy(labels,logits)

   # convert from one_hot to sparse
   logits = tf.argmax(logits, axis=1)
   return logits, loss
