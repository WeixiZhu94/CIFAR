import tensorflow as tf
def _logsumexp(t, mask):
   logit = tf.gather_nd(t, mask)
   max = tf.reduce_max(logit, 0, keep_dims=True)
   epsilon = tf.log(tf.reduce_sum(tf.exp(logit - max), 0, keep_dims=True))
   return max + epsilon

# 100 -> 20
def _cat1(labels):
   with tf.variable_scope('cat_1_label'):
      table_0  = tf.constant([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
      table_1  = tf.constant([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
      table_2  = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
      table_3  = tf.constant([0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      table_4  = tf.constant([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      table_5  = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
      table_6  = tf.constant([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
      table_7  = tf.constant([0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      table_8  = tf.constant([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0])
      table_9  = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      table_10 = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      table_11 = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      table_12 = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      table_13 = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
      table_14 = tf.constant([0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
      table_15 = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
      table_16 = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      table_17 = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
      table_18 = tf.constant([0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
      table_19 = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
      A = tf.stack([table_0,table_1,table_2,table_3,table_4,table_5,table_6,table_7,table_8,table_9,table_10,table_11,table_12,table_13,table_14,table_15,table_16,table_17,table_18,table_19], axis=0)
      A = tf.transpose(A)
      B = tf.matmul(tf.one_hot(labels, 100, 1, 0, axis=-1), A)
   return tf.argmax(B, axis=1)

#20 -> 9
def _cat2(labels):
   with tf.variable_scope('cat_2_label'):
      table_0 = tf.constant([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      table_1 = tf.constant([0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0])
      table_2 = tf.constant([0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,1,0,0,0])
      table_3 = tf.constant([0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      table_4 = tf.constant([0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
      table_5 = tf.constant([0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0])
      table_6 = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
      table_7 = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
      table_8 = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1])
      A = tf.stack([table_0, table_1, table_2, table_3, table_4, table_5, table_6, table_7, table_8], axis=0)
      A = tf.transpose(A)
      B = tf.matmul(tf.one_hot(labels, 20, 1, 0, axis=-1), A)
   return tf.argmax(B, axis=1)

#9 -> 4
def _cat3(labels):
   with tf.variable_scope('cat_3_label'):
      table_0 = tf.constant([1,0,1,0,0,0,1,0,0])
      table_1 = tf.constant([0,1,0,0,0,0,0,0,0])
      table_2 = tf.constant([0,0,0,1,0,0,0,1,0])
      table_3 = tf.constant([0,0,0,0,1,1,0,0,1])
      A = tf.stack([table_0, table_1, table_2, table_3], axis=0)
      A = tf.transpose(A)
      B = tf.matmul(tf.one_hot(labels, 9, 1, 0, axis=-1), A)
   return tf.argmax(B, axis=1)

#4 -> 3
def _cat4(labels):
   with tf.variable_scope('cat_4_label'):
      table_0 = tf.constant([1,1,0,0])
      table_1 = tf.constant([0,0,1,0])
      table_2 = tf.constant([0,0,0,1])
      A = tf.stack([table_0, table_1, table_2], axis=0)
      A = tf.transpose(A)
      B = tf.matmul(tf.one_hot(labels, 4, 1, 0, axis=-1), A)
   return tf.argmax(B, axis=1)

def _cat5(labels):
   with tf.variable_scope('cat_5_label'):
      table_0 = tf.constant([1,1,0])
      table_1 = tf.constant([0,0,1])
      A = tf.stack([table_0, table_1], axis=0)
      A = tf.transpose(A)
      B = tf.matmul(tf.one_hot(labels, 3, 1, 0, axis=-1), A)
   return tf.argmax(B, axis=1)

def _cat1_logits(logits):
   with tf.variable_scope('cat_1_logsumexp'):
      t = tf.transpose(logits)
      mask_0 = _logsumexp(t, tf.constant([[4],[30],[55],[72],[95]]))
      mask_1 = _logsumexp(t, tf.constant([[1],[32],[67],[73],[91]]))
      mask_2 = _logsumexp(t, tf.constant([[54],[62],[70],[82],[92]]))
      mask_3 = _logsumexp(t, tf.constant([[9],[10],[16],[28],[61]]))
      mask_4 = _logsumexp(t, tf.constant([[0],[51],[53],[57],[83]]))
      mask_5 = _logsumexp(t, tf.constant([[22],[39],[40],[86],[87]]))
      mask_6 = _logsumexp(t, tf.constant([[5],[20],[25],[84],[94]]))
      mask_7 = _logsumexp(t, tf.constant([[6],[7],[14],[18],[24]]))
      mask_8 = _logsumexp(t, tf.constant([[3],[42],[43],[88],[97]]))
      mask_9 = _logsumexp(t, tf.constant([[12],[17],[37],[68],[76]]))
      mask_10 = _logsumexp(t, tf.constant([[23],[33],[49],[60],[71]]))
      mask_11 = _logsumexp(t, tf.constant([[15],[19],[21],[31],[38]]))
      mask_12 = _logsumexp(t, tf.constant([[34],[63],[64],[66],[75]]))
      mask_13 = _logsumexp(t, tf.constant([[26],[45],[77],[79],[99]]))
      mask_14 = _logsumexp(t, tf.constant([[2],[11],[35],[46],[98]]))
      mask_15 = _logsumexp(t, tf.constant([[27],[29],[44],[78],[93]]))
      mask_16 = _logsumexp(t, tf.constant([[36],[50],[65],[74],[80]]))
      mask_17 = _logsumexp(t, tf.constant([[47],[52],[56],[59],[96]]))
      mask_18 = _logsumexp(t, tf.constant([[8],[13],[48],[58],[90]]))
      mask_19 = _logsumexp(t, tf.constant([[41],[69],[81],[85],[89]]))
      logits = tf.transpose(tf.concat([mask_0,mask_1,mask_2,mask_3,mask_4,mask_5,mask_6,mask_7,mask_8,mask_9,mask_10,mask_11,mask_12,mask_13,mask_14,mask_15,mask_16,mask_17,mask_18,mask_19], axis=0))
   return logits

def _cat2_logits(logits):
   with tf.variable_scope('cat_2_logsumexp'):
      t = tf.transpose(logits)
      mask_0 = _logsumexp(t, tf.constant([[0],[1]]))
      mask_1 = _logsumexp(t, tf.constant([[7],[13]]))
      mask_2 = _logsumexp(t, tf.constant([[8],[11],[12],[14],[16]]))
      mask_3 = _logsumexp(t, tf.constant([[2],[4]]))
      mask_4 = _logsumexp(t, tf.constant([[3],[5],[6]]))
      mask_5 = _logsumexp(t, tf.constant([[9],[10]]))
      mask_6 = _logsumexp(t, tf.constant([[15]]))
      mask_7 = _logsumexp(t, tf.constant([[17]]))
      mask_8 = _logsumexp(t, tf.constant([[18],[19]]))
      logits = tf.transpose(tf.concat([mask_0,mask_1,mask_2,mask_3,mask_4,mask_5,mask_6,mask_7,mask_8], axis=0))
   return logits

def _cat3_logits(logits):
   with tf.variable_scope('cat_3_logsumexp'):
      t = tf.transpose(logits)
      mask_0 = _logsumexp(t, tf.constant([[0],[2],[6]]))
      mask_1 = _logsumexp(t, tf.constant([[1]]))
      mask_2 = _logsumexp(t, tf.constant([[3],[7]]))
      mask_3 = _logsumexp(t, tf.constant([[4],[5],[8]]))
      logits = tf.transpose(tf.concat([mask_0,mask_1,mask_2,mask_3], axis=0))
   return logits

def _cat4_logits(logits):
   with tf.variable_scope('cat_4_logsumexp'):
      t = tf.transpose(logits)
      mask_0 = _logsumexp(t, tf.constant([[0],[1]]))
      mask_1 = _logsumexp(t, tf.constant([[2]]))
      mask_2 = _logsumexp(t, tf.constant([[3]]))
      logits = tf.transpose(tf.concat([mask_0,mask_1,mask_2], axis=0))
   return logits

def _cat5_logits(logits):
   with tf.variable_scope('cat_5_logsumexp'):
      t = tf.transpose(logits)
      mask_0 = _logsumexp(t, tf.constant([[0],[1]]))
      mask_1 = _logsumexp(t, tf.constant([[2]]))
      logits = tf.transpose(tf.concat([mask_0,mask_1], axis=0))
   return logits