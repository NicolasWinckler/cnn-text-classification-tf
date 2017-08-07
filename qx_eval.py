#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import pandas
import os
import data_helpers
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import yaml
import re
from prettytable import PrettyTable
def get_numbers_from_filename(filename):
    return re.search(r'\d+', filename).group(0)



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


# Parameters
# ==================================================

# Data Parameters

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("config", "qx_config.yml", "yaml config file")
#tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


config_file = FLAGS.config
with open(config_file, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

datasets = None


# ====================================================================================================
# Data Preparation
# ====================================================================================================
#
# Load training data
#
print("Loading test data...")

dataset_name = "localdata"
datasets = data_helpers.get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["testset_path"],
                                                     categories=cfg["datasets"][dataset_name]["categories"],
                                                     shuffle=False,#shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                     random_state=cfg["datasets"][dataset_name]["random_state"])
x_raw, y_test = data_helpers.load_data_labels(datasets)

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# ====================================================================================================
# Evaluation
# ====================================================================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_probabilities = None

        for x_test_batch in batches:
            batch_predictions_scores = sess.run([predictions, scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
            probabilities = softmax(batch_predictions_scores[1])
            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities, probabilities])
            else:
                all_probabilities = probabilities



y_test = np.argmax(y_test, axis=1)

# Print evaluation summary on splitted document
# ====================================================================================================

if y_test is None:
    print("[ERROR] observed class array, y_test, is empty. Program will now exit ")
    exit()


def get_summary_header(obs, pred):
    correct_predictions = float(sum(pred == obs))
    p = PrettyTable()
    p.field_names = ["correct predictions", "incorrect predictions", "Test set sample size", "Accuracy"]
    acc = "{:g}".format(correct_predictions/float(len(obs)))
    test_size = str(len(pred))
    cor_pred_size = str(int(correct_predictions))
    incor_pred_size = str(sum(pred != obs))
    p.add_row([cor_pred_size, incor_pred_size, test_size, acc])

    p.align["correct predictions"] = "c"
    p.align["incorrect predictions"] = "c"
    p.align["Test set sample size"] = "c"
    p.align["Accuracy"] = "c"
    return p.get_string(header=True, border=True)


def confusion_matrix_string(obs, pred, target_names):
    correct_predictions = float(sum(pred == obs))
    p = PrettyTable()
    field_names = []
    for categ in target_names:
        field_name = categ.split('_')
        field_names.append(field_name[0])
    p.field_names = field_names

    confmat = metrics.confusion_matrix(obs, pred)
    for row in confmat:
        p.add_row(row)

    field_len = len(field_names)
    p.add_column("cluster",field_names)
    p.align["cluster"] = "r"
    return p.get_string(header=True, border=False)

table_txt = confusion_matrix_string(y_test, all_predictions, datasets['target_names'])

print(table_txt)

report_summary1 = "========================= Evaluation summary of splitted documents =========================\n"
report_summary1 += "\n" 
report_summary1 += get_summary_header(y_test, all_predictions)
report_summary1 += "\n \n"
reportsplitdoc = metrics.classification_report(y_test, all_predictions, target_names=datasets['target_names'])
report_summary1 += reportsplitdoc
report_summary1 += "\n \n"
report_summary1 += str(metrics.confusion_matrix(y_test, all_predictions))
print(report_summary1)


# parse sklearn report 
report_data = []
lines = reportsplitdoc.split('\n')
for line in lines[2:-3]:
    row = {}
    row_data = line.split() 
    row['class'] = row_data[0]
    row['precision'] = float(row_data[1])
    row['recall'] = float(row_data[2])
    row['f1_score'] = float(row_data[3])
    row['support'] = float(row_data[4])
    report_data.append(row)



# Save the evaluation to a csv
# ====================================================================================================

predictions_human_readable = np.column_stack((np.array(x_raw),
                                              [int(prediction) for prediction in all_predictions],
                                              [ "{}".format(probability) for probability in all_probabilities]))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)



# ====================================================================================================
# Reconstruction of documents
# ====================================================================================================

idx2file_map = {}
#get map: key=file, value splited doc index array 
data_helpers.get_file_mapping(cfg['mappingfile'], invert_mapper = idx2file_map)

# for each splitted file, parse the filename to get the index
# and hence get the corresponding original source file name via the map
# Then create a new map that map source files to splitted doc triplet [idx, pred, obs]
# ====================================================================================================

y_final_pred_map = {}
for i in range(len(datasets['filenames'])):
    file = datasets['filenames'][i]
    label = datasets['target'][i]
    # remove path
    path, filename = os.path.split(file)
    path, categ = os.path.split(path)
    # get index from splitted file
    idx = int(get_numbers_from_filename(filename))
    # get source file name from splitted doc idx
    sourcefile = idx2file_map[idx]
    pred = all_predictions[i]
    # fill map
    if sourcefile not in y_final_pred_map:
        y_final_pred_map[sourcefile] = list()
    y_final_pred_map[sourcefile].append([idx,pred,label])


# vote system to labelize the reconstructed document from splitted doc predictions
# find during the process the misclassified reconstructed document
# ====================================================================================================


y_final_test = []
y_final_pred = []
misclassified_files = []

Normalize = 0.0
for r in report_data:
    Normalize = Normalize + r['support']

for file, val in y_final_pred_map.items():
    hist = {}
    # fill the new y_test with observed class
    y_final_test.append(int(val[0][2]))
    # get predicted class, and fill an histogram
    for v in val:
        if v[1] not in hist:
            hist[v[1]] = 0
        hist[v[1]] = hist[v[1]] + 1

    # apply weight on prediction (not yet used, doesn t improve accuracy)
    use_weight = False
    if use_weight:
        weight = 1.0
        for categnb, freq in hist.items():
            idx = int(categnb)
            categstr = datasets['target_names'][idx]
            for r in report_data:
                if categstr in r['class']:
                    weight = r['support']
            hist[categnb] = freq*weight/Normalize
    # get key with max values in hist 
    vals = list(hist.values())
    keys = list(hist.keys())
    results = keys[vals.index(max(vals))]
    # fill the new y_test with predicted class
    y_final_pred.append(int(results))
    # find misclassified files
    if(results != int(val[0][2])):
        path, file = os.path.split(file)
        path, categ = os.path.split(path)
        file = categ + "/" + file
        obs=datasets['target_names'][int(val[0][2])]
        pred=datasets['target_names'][int(results)]
        misclassified_files.append([file,obs,pred])




y_final_test = np.array(y_final_test)
y_final_pred = np.array(y_final_pred)

#print("y_final_test = {}".format(y_final_test))
#print("y_final_pred = {}".format(y_final_pred))


# format, print and save misclassified files
# ====================================================================================================

p = PrettyTable()
p.field_names = ["Source File", "Observed label", "Predicted label"]
for row in misclassified_files:
   p.add_row(row)

p.align["Source File"] = "l"
p.align["Observed label"] = "c"
p.align["Predicted label"] = "c"

table_txt = "\n \n========================= Misclassified documents =========================\n \n"
table_txt += p.get_string(header=True, border=True)
print(table_txt)

# save to file
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "misclassified.txt")
print("Saving miclassified document to {0}\n \n".format(out_path))
with open(out_path, 'w') as f:
    f.write(table_txt)



# print evaluation summary for rectonstructed documents
# ====================================================================================================

if y_final_test is None:
    print("[ERROR] observed class array, y_final_test, is empty. Program will now exit ")
    exit()
correct_predictions = float(sum(y_final_pred == y_final_test))
###
report_summary2 = "========================= Evaluation summary of reconstructed documents =========================\n"
report_summary2 += "\n \n" 
report_summary2 += "- Total number of source test examples: {}\n".format(len(y_test))
report_summary2 += "- Accuracy on source documents: {:g}\n".format(correct_predictions/float(len(y_test)))
report_summary2 += "\n \n"
report_summary2 += metrics.classification_report(y_final_test, y_final_pred, target_names=datasets['target_names'])
report_summary2 += "\n \n"
report_summary2 += str(metrics.confusion_matrix(y_final_test, y_final_pred))
print(report_summary2)

###
report_summary = report_summary1 + "\n \n"
report_summary += report_summary2
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "eval_summary.txt")
print("Saving evaluation summary to {0}".format(out_path))
with open(out_path, 'w') as f:
    f.write(report_summary)

# Save the evaluation to a csv
# predictions_human_readable = np.column_stack((np.array(x_raw),
#                                               [int(prediction) for prediction in y_final_pred],
#                                               [ "{}".format(probability) for probability in all_probabilities]))
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
# print("Saving evaluation to {0}".format(out_path))
# with open(out_path, 'w') as f:
#     csv.writer(f).writerows(predictions_human_readable)

