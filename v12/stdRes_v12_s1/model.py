import os

import tensorflow as tf
import tensorflow.contrib.slim as slim\

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'train.tfrecords'
TEST_FILE = 'test.tfrecords'

regularizer = slim.l2_regularizer(0.0005)

def _cat1(labels):
   table1 = tf.constant([1,0,0,0,0,0])
   table2 = tf.constant([0,1,0,0,0,0])
   table3 = tf.constant([0,0,0,0,1,0])
   table4 = tf.constant([0,0,0,1,0,0])
   table5 = tf.constant([0,0,0,1,0,0])
   table6 = tf.constant([0,0,0,1,0,0])
   table7 = tf.constant([0,0,0,0,0,1])
   table8 = tf.constant([0,0,0,1,0,0])
   table9 = tf.constant([0,0,1,0,0,0])
   table0 = tf.constant([0,1,0,0,0,0])
   A = tf.stack([table1, table2, table3, table4, table5, table6, table7, table8, table9, table0], axis=0)
   one_hot = tf.one_hot(labels, 10, 1, 0, axis=-1)
   return tf.argmax(tf.matmul(one_hot, A), axis=1)

def _cat2(labels):
   table1 = tf.constant([1,1,0,0,0,0,0,0,1,1])
   table2 = tf.constant([0,0,1,1,1,1,1,1,0,0])
   A = tf.transpose(tf.stack([table1, table2], axis=0))
   one_hot = tf.one_hot(labels, 10, 1, 0, axis=-1)
   return tf.argmax(tf.matmul(one_hot, A), axis=1)

def _logsumexp(t, mask):
   logit = tf.gather_nd(t, mask)
   max = tf.reduce_max(logit, 0, keep_dims=True)
   epsilon = tf.log(tf.reduce_sum(tf.exp(logit - max), 0, keep_dims=True))
   return max + epsilon

def _cat1_logits(logits):
   with tf.variable_scope('cat_1'):
      mask0 = tf.constant([[0]])
      mask1 = tf.constant([[1], [9]])
      mask2 = tf.constant([[8]])
      mask3 = tf.constant([[3], [4], [5], [7]])
      mask4 = tf.constant([[2]])
      mask5 = tf.constant([[6]])
      t = tf.transpose(logits)
      logits = tf.transpose(tf.concat([_logsumexp(t, mask0), _logsumexp(t, mask1), _logsumexp(t, mask2), _logsumexp(t, mask3), _logsumexp(t, mask4), _logsumexp(t, mask5)], axis=0))
   return logits

def _cat2_logits(logits):
   with tf.variable_scope('cat_2'):
      mask0 = tf.constant([[0], [1], [2]])
      mask1 = tf.constant([[3], [4], [5]])
      t = tf.transpose(logits)
      logits = tf.transpose(tf.concat([_logsumexp(t, mask0), _logsumexp(t, mask1)], axis=0))
   return logits

def _residual(net, in_filter, out_filter, prefix):
   # ori_net : not activated; net -> BN -> RELU
   with tf.variable_scope(prefix+'_pre_act'):
      ori_net = net
      net = slim.layers.batch_norm(net)
      net = tf.nn.relu(net)
   with tf.variable_scope(prefix+'_residual'):
      # net -> Weight -> BN -> RELU
      net = slim.layers.conv2d(net, out_filter, [3,3], scope='conv_1', normalizer_fn=slim.layers.batch_norm, biases_regularizer=regularizer, weights_regularizer=regularizer)
      # net -> Weight
      net = slim.layers.conv2d(net, out_filter, [3,3], scope='conv_2', activation_fn=None, biases_regularizer=regularizer, weights_regularizer=regularizer)
   with tf.variable_scope(prefix+'_res_add'):
      if in_filter != out_filter:
         ori_net = tf.nn.avg_pool(ori_net, [1,1,1,1], [1,1,1,1], 'VALID')
         ori_net = tf.pad(ori_net, [[0,0],[0,0],[0,0],[(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      net += ori_net
   return net

def _si_conv(net, in_filter, out_filter, prefix):
   net = slim.layers.conv2d(net, out_filter, [3,3], scope=prefix + 'conv_1', normalizer_fn=slim.layers.batch_norm, biases_regularizer=regularizer, weights_regularizer=regularizer)
   return net

def _bi_conv(net, in_filter, out_filter, prefix):
   net = slim.layers.conv2d(net, out_filter, [3,3], scope=prefix + 'conv_1', normalizer_fn=slim.layers.batch_norm, biases_regularizer=regularizer, weights_regularizer=regularizer)
   net = slim.layers.conv2d(net, out_filter, [3,3], scope=prefix + 'conv_2', normalizer_fn=slim.layers.batch_norm, biases_regularizer=regularizer, weights_regularizer=regularizer)
   return net

def network(net, labels):

   net = _si_conv(net, 8, 8, 'res_init')
   
   net = _residual(net, 8, 8, 'unit_8_1')
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_1')

   net = _residual(net, 8, 16, 'unit_16_1')
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_2')

   with tf.variable_scope('res_last'):
      net = slim.layers.batch_norm(net)
      net = tf.nn.relu(net)
      net = tf.reduce_mean(net, [1,2])

   logits = slim.layers.fully_connected(net, 10, activation_fn=None, scope='logits',biases_regularizer=regularizer, weights_regularizer=regularizer)
   #logits_cat1 = slim.layers.fully_connected(net, 6, activation_fn=None, scope='logits_cat1',biases_regularizer=regularizer, weights_regularizer=regularizer)
   #logits_cat2 = slim.layers.fully_connected(net, 2, activation_fn=None, scope='logits_cat2',biases_regularizer=regularizer, weights_regularizer=regularizer)
   logits_cat1 = _cat1_logits(logits)
   logits_cat2 = _cat2_logits(logits_cat1)
   
   labels_cat1 = _cat1(labels)
   labels_cat2 = _cat2(labels)

   loss = tf.losses.sparse_softmax_cross_entropy(labels,logits)
   loss_cat1 = tf.losses.sparse_softmax_cross_entropy(labels_cat1, logits_cat1)
   loss_cat2 = tf.losses.sparse_softmax_cross_entropy(labels_cat2, logits_cat2)
   return logits, logits_cat1, logits_cat2, loss, loss_cat1, loss_cat2, labels_cat1, labels_cat2

