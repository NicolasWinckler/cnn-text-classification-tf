#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import yaml


# helper function 


def write_param_to_tf_board(out_directory,flagscontainer, text_list = []):
    for attr, value in sorted(flagscontainer.__flags.items()):
        ele_ar = [str(attr.upper()),str(value)]
        text_list.append(ele_ar)

    summary_op = tf.summary.text('Setting', tf.convert_to_tensor(text_list))
    summary_writer = tf.summary.FileWriter(out_directory, sess.graph)
    text_list = sess.run(summary_op)
    summary_writer.add_summary(text_list, 0)
    summary_writer.flush()
    summary_writer.close()


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_boolean("enable_word_embeddings", False, "Enable/disable the word embedding (default: True)")
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.4, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string("config", "qx_config.yml", "yaml config file")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



config_file = FLAGS.config
with open(config_file, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


#dataset_name = cfg["datasets"]["default"]
dataset_name = "localdata" #cfg["datasets"]["default"]
if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
    embedding_name = cfg['word_embeddings']['default']
    embedding_dimension = cfg['word_embeddings'][embedding_name]['dimension']
else:
    embedding_name = "internal w2v"
    embedding_dimension = FLAGS.embedding_dim

# ====================================================================================================
# Data Preparation
# ====================================================================================================
#
# Load training data
#
print("Loading training data...")

dataset_name = "localdata"
data_dir = os.path.abspath(cfg["datasets"][dataset_name]["container_path"])
datasets_train = data_helpers.get_datasets_localdata(container_path=os.path.join(data_dir, "train"),
                                                     categories=cfg["datasets"][dataset_name]["categories"],
                                                     shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                     random_state=cfg["datasets"][dataset_name]["random_state"])
x_raw_train, y_raw_train = data_helpers.load_data_labels(datasets_train)

# ==================================================
#
# Load test data
#

print("Loading test data...")
datasets_test = data_helpers.get_datasets_localdata(container_path=os.path.join(data_dir, "test"),
                                                     categories=cfg["datasets"][dataset_name]["categories"],
                                                     shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                     random_state=cfg["datasets"][dataset_name]["random_state"])
x_raw_test, y_raw_test = data_helpers.load_data_labels(datasets_test)



# ==================================================
#
# Build vocabulary
#
# concatenate to build a global vocabulary
x_raw = np.concatenate((x_raw_train, x_raw_test),axis=0)
y_raw = np.concatenate((y_raw_train, y_raw_test),axis=0)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_raw])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

# Note: to get the word -> id map:
# vocab_dict = vocab_processor.vocabulary_._mapping

x_ordered_train = np.array(list(vocab_processor.fit_transform(x_raw_train)))
x_ordered_test = np.array(list(vocab_processor.fit_transform(x_raw_test)))


# ==================================================
# Randomly shuffle train data
np.random.seed(10)
shuffle_train_indices = np.random.permutation(np.arange(len(y_raw_train)))
x_train = x_ordered_train[shuffle_train_indices]
y_train = y_raw_train[shuffle_train_indices]


# ==================================================
# Randomly shuffle test data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_raw_test)))
x_dev = x_ordered_test[shuffle_indices]
y_dev = y_raw_test[shuffle_indices]

class_number = len(y_raw[0])
vocab_size = len(vocab_processor.vocabulary_)
y_train_size = len(y_train)
y_dev_size = len(y_dev)


print("Max document length in training + test set : {:d}".format(max_document_length))
print("Number of distinct labels in training + test set : {:d}".format(class_number))
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/test split: {:d}/{:d}".format(y_train_size, y_dev_size))


# ====================================================================================================
# Training
# ====================================================================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=embedding_dimension,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        #timestamp = str(int(time.time()))
        timestamp = time.strftime("%Y-%m-%d-%a-%H%M%S", time.localtime())
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        input_param = []
        input_param.append(["DATA SET NAME",str(dataset_name)])
        input_param.append(["LABELED CLASS NB",str(class_number)])
        input_param.append(["MAX NB OF WORDS/DOC",str(max_document_length)])
        input_param.append(["TRAINING SET SIZE",str(y_train_size)])
        input_param.append(["TEST SET SIZE",str(y_dev_size)])
        input_param.append(["CLUSTERS PATH",str(cfg["datasets"][dataset_name]["container_path"])])
        input_param.append(["CLUSTERS NAMES",str(cfg["datasets"][dataset_name]["categories"])])
        input_param.append(["EMBEDDINGS METHOD",str(embedding_name)])
        input_param.append(["EMBEDDINGS DIMENSION",str(embedding_dimension)])
        
        write_param_to_tf_board(out_dir,FLAGS,input_param)
        
        
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        #inputpar_summary = tf.summary.text('input_param', tf.convert_to_tensor('hello world'))

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
            vocabulary = vocab_processor.vocabulary_
            initW = None
            if embedding_name == 'word2vec':
                # load embedding vectors from the word2vec
                print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
                initW = data_helpers.load_embedding_vectors_word2vec(vocabulary,
                                                                     cfg['word_embeddings']['word2vec']['path'],
                                                                     cfg['word_embeddings']['word2vec']['binary'])
                print("word2vec file has been loaded")
            elif embedding_name == 'glove':
                # load embedding vectors from the glove
                print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
                initW = data_helpers.load_embedding_vectors_glove(vocabulary,
                                                                  cfg['word_embeddings']['glove']['path'],
                                                                  embedding_dimension)
                print("glove file has been loaded\n")
            sess.run(cnn.W.assign(initW))

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, best_acc, best_loss, current_step, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if accuracy > best_acc and accuracy > 0.6:
                best_acc = accuracy

            if loss < best_loss:
                loss = best_loss
            
            if writer:
                writer.add_summary(summaries, step)
            
            return accuracy, best_acc, best_loss

        # Generate batches
        current_acc = 0.0
        best_acc = 0.0
        best_loss = 1000000000.0
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                current_acc, best_acc, best_loss = dev_step(x_dev, y_dev, best_acc, best_loss ,current_step, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                #if current_acc >= best_acc and current_acc > 0.6:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


cfg_out = out_dir + "/" + "qx_config.yml"

with open(cfg_out, 'w') as outfile:
    yaml.dump(cfg, outfile, default_flow_style=False)
