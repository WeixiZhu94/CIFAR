import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys, os
from model import network
from cifar_input import build_input

flags = tf.app.flags
flags.DEFINE_string('mode', 'train',
                    'train as default')
flags.DEFINE_float('lrn', 0.1, 'learning_rate')
FLAGS = flags.FLAGS

def report():
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
            TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

def train(log_dir, lrn):
    print('Saving inside logdir: ' + log_dir)
    images, labels = build_input('cifar10', 128, 'train')
    logits, loss = network(images, labels)
    
    report()

    tf.summary.scalar('losses/loss', loss)
    tf.summary.scalar('error', 1 - slim.metrics.accuracy(logits, tf.to_int64(labels)))

    optimizer = tf.train.MomentumOptimizer(learning_rate=lrn, momentum=0.9)
    tf.summary.scalar('learning_rate', lrn)

    #phrase 1
    #total_loss = loss_cat2
    #phrase 2
    #total_loss = (loss_cat2 + loss_cat1) / 2
    #phrase 3
    #total_loss = (loss_cat2 + loss_cat1 + loss) / 3
    #phrase 4
    total_loss = loss
    train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

    slim.learning.train(train_op, log_dir, save_summaries_secs=20, save_interval_secs=20)


def eval(log_dir, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = log_dir

    images, labels = build_input('cifar10', 10000, 'test')
    logits, loss = network(images, labels)
  
    tf.summary.scalar('losses/loss', loss)
    tf.summary.scalar('error', 1 - slim.metrics.accuracy(logits, tf.to_int64(labels)))

    # Evaluate every 30 seconds
    slim.evaluation.evaluation_loop(
        '',
        checkpoint_dir,
        log_dir,
        num_evals=1,
        summary_op=tf.summary.merge_all(),
        eval_interval_secs=20,
        max_number_of_evaluations= 10000000,
        )

def main(mode, lrn):
    prefix = os.path.relpath('.', '../..')
    log_prefix = '../../log/' + prefix
    if mode == 'train':
        train(log_prefix + '/train', lrn)
    else:
        eval(log_prefix + '/eval', log_prefix + '/train')

if __name__ == '__main__':
    main(FLAGS.mode,
         FLAGS.lrn)