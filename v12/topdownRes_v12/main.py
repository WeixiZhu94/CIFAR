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
    logits, logits_cat1, logits_cat2, loss, loss_cat1, loss_cat2, labels_cat1, labels_cat2 = network(images, labels)
    
    report()

    tf.summary.scalar('losses/loss', loss)
    tf.summary.scalar('losses/loss_cat1', loss_cat1)
    tf.summary.scalar('losses/loss_cat2', loss_cat2)

    logits = tf.argmax(logits, axis=1)
    logits_cat1 = tf.argmax(logits_cat1, axis=1)
    logits_cat2 = tf.argmax(logits_cat2, axis=1)

    tf.summary.scalar('accuracy', slim.metrics.accuracy(logits, tf.to_int64(labels)))
    tf.summary.scalar('accuracy_cat_1', slim.metrics.accuracy(logits_cat1, tf.to_int64(labels_cat1)))
    tf.summary.scalar('accuracy_cat_2', slim.metrics.accuracy(logits_cat2, tf.to_int64(labels_cat2)))

    optimizer = tf.train.MomentumOptimizer(0.9, lrn)
    tf.summary.scalar('learning_rate', lrn)
    total_loss = loss_cat2
    train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

    slim.learning.train(train_op, log_dir, save_summaries_secs=20, save_interval_secs=20)


def eval(log_dir):

    images, labels = build_input('cifar10', 10000, 'test')
    logits, logits_cat1, logits_cat2, loss, loss_cat1, loss_cat2, labels_cat1, labels_cat2 = network(images, labels)
  
    tf.summary.scalar('losses/loss', loss)
    tf.summary.scalar('losses/loss_cat1', loss_cat1)
    tf.summary.scalar('losses/loss_cat2', loss_cat2)

    logits = tf.argmax(logits, axis=1)
    logits_cat1 = tf.argmax(logits_cat1, axis=1)
    logits_cat2 = tf.argmax(logits_cat2, axis=1)

    tf.summary.scalar('accuracy', slim.metrics.accuracy(logits, tf.to_int64(labels)))
    tf.summary.scalar('accuracy_cat_1', slim.metrics.accuracy(logits_cat1, tf.to_int64(labels_cat1)))
    tf.summary.scalar('accuracy_cat_2', slim.metrics.accuracy(logits_cat2, tf.to_int64(labels_cat2)))

    # These are streaming metrics which compute the "running" metric,
    # e.g running accuracy
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
        'accuracies/accuracy': slim.metrics.streaming_accuracy(logits, labels),
        'accuracies/accuracy_cat_1': slim.metrics.streaming_accuracy(logits_cat1, labels_cat1),
        'accuracies/accuracy_cat_2': slim.metrics.streaming_accuracy(logits_cat2, labels_cat2),
    })

    # Define the streaming summaries to write:
    for metric_name, metric_value in metrics_to_values.items():
        tf.summary.scalar(metric_name, metric_value)

    # Evaluate every 30 seconds
    slim.evaluation.evaluation_loop(
        '',
        log_dir,
        log_dir,
        num_evals=1,
        eval_op=list(metrics_to_updates.values()),
        summary_op=tf.summary.merge_all(),
        eval_interval_secs=30,
        max_number_of_evaluations=10000000,
        )

def main(mode, lrn):
    prefix = os.path.relpath('.', '../..')
    log_prefix = '../../log/' + prefix
    if mode == 'train':
        train(log_prefix + '/train', lrn)
    else:
        eval(log_prefix + '/eval')

if __name__ == '__main__':
    main(FLAGS.mode,
         FLAGS.lrn)