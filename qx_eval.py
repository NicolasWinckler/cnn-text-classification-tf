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
def get_numbers_from_filename(filename):
    return re.search(r'\d+', filename).group(0)



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

with open("qx_config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# Parameters
# ==================================================

# Data Parameters

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
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

# Evaluation
# ==================================================
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





print("all_predictions = {} ".format(all_predictions))
print("all_predictions size = {} ".format(len(all_predictions)))
print("y_test = {} ".format(y_test))
print("y_test size = {} ".format(len(y_test)))

y_test = np.argmax(y_test, axis=1)
print("y_test = {} ".format(y_test))
print("y_test size = {} ".format(len(y_test)))


# Print accuracy if y_test is defined

if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy on splitted documents: {:g}".format(correct_predictions/float(len(y_test))))
    reportsplitdoc = metrics.classification_report(y_test, all_predictions, target_names=datasets['target_names'])
    print(reportsplitdoc)
    print(metrics.confusion_matrix(y_test, all_predictions))



report_data = []
lines = reportsplitdoc.split('\n')
#print(lines)
for line in lines[2:-3]:
    row = {}
    row_data = line.split() 
    row['class'] = row_data[0]
    row['precision'] = float(row_data[1])
    row['recall'] = float(row_data[2])
    row['f1_score'] = float(row_data[3])
    row['support'] = float(row_data[4])
    report_data.append(row)

#print("REPORT = {}".format(report_data))


# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw),
                                              [int(prediction) for prediction in all_predictions],
                                              [ "{}".format(probability) for probability in all_probabilities]))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)




idx2file_map = {}
file2idx_map = {}
data_helpers.get_file_mapping(cfg['mappingfile'], file2idx_map, idx2file_map)

#for file in file2idx_map:
#    print("file = {}".format(file))

y_final_pred_map = {}
print("files : {}".format(datasets['filenames']))
#dataset['target_names']
#for file in datasets['filenames']:
for i in range(len(datasets['filenames'])):
    file = datasets['filenames'][i]
    label = datasets['target'][i]
    #print("i={}".format(i))
    #print(all_predictions[i])
    path, filename = os.path.split(file)
    path, categ = os.path.split(path)
    idx = int(get_numbers_from_filename(filename))
    #print(idx)
    sourcefile = idx2file_map[idx]
    #print(sourcefile)
    pred = all_predictions[i]
    if sourcefile not in y_final_pred_map:
        y_final_pred_map[sourcefile] = list()
    y_final_pred_map[sourcefile].append([idx,pred,label])

y_final_test = []
y_final_pred = []
misclassified_files = {}

Normalize = 0.0
for r in report_data:
    Normalize = Normalize + r['support']

for file, val in y_final_pred_map.items():
    #print(file)
    #print(val)
    hist = {}
    y_final_test.append(int(val[0][2]))
    for v in val:
        if v[1] not in hist:
            hist[v[1]] = 0
        hist[v[1]] = hist[v[1]] + 1

    # weight not yet used 
    use_weight = False
    if use_weight:
        weight = 1.0
        for categnb, freq in hist.items():
            idx = int(categnb)
            categstr = datasets['target_names'][idx]
            for r in report_data:
                if categstr in r['class']:
                    weight = r['support']
            hist[categnb] = freq*weight
    # get key with max values in hist 
    vals = list(hist.values())
    keys = list(hist.keys())
    results = keys[vals.index(max(vals))]
    y_final_pred.append(int(results))
    if(results != int(val[0][2])):
        path, file = os.path.split(file)
        path, categ = os.path.split(path)
        file = categ + "/" + file
        misclassified_files[file] = list([int(val[0][2]),int(results)])






y_final_test = np.array(y_final_test)
y_final_pred = np.array(y_final_pred)


print("y_final_test = {}".format(y_final_test))
print("y_final_pred = {}".format(y_final_pred))

import pprint

print("misclassified files :")
misclassified_files_array = []
temp_array = []
for file, val in misclassified_files.items():
    obs=datasets['target_names'][val[0]]
    pred=datasets['target_names'][val[1]]
    print("files = {}   obs: {}   pred: {} ".format(file,obs,pred))
    misclassified_files_array.append([file,obs,pred])
    temp_array.append([file,obs,pred])



# #print(pandas.DataFrame(data=misclassified_files_array))
# width = max(len(cn) for cn in misclassified_files)
# headers = ["file", "obs", "pred"]
# fmt = '%% %ds' % width  # first column: class name
# fmt += '  '
# fmt += ' '.join(['% 9s' for _ in headers])
# fmt += '\n'
# headers = [""] + headers
# report = fmt % tuple(headers)
# report += '\n'

# for val in misclassified_files_array:
#     report += val[0] + "   " + val[1] + "  " +val[2] +"\n"

# #     values = [file]
# #     for v in val:
# #         values += [" {} ".format(v)]
# #     report += fmt % tuple(values)

# print(report)


# from prettytable import PrettyTable

# x = [["A", "B"], ["C", "D"]]

#{file:[obs,pred]}
p = PrettyTable()
for row in temp_array:
   p.add_row(row)

p.align = "l"
print(p.get_string(header=False, border=False))


# Print accuracy if y_test is defined
if y_final_test is not None:
    correct_predictions = float(sum(y_final_pred == y_final_test))
    print("Total number of test examples: {}".format(len(y_final_test)))
    print("Accuracy on source documents: {:g}".format(correct_predictions/float(len(y_final_test))))
    print(metrics.classification_report(y_final_test, y_final_pred, target_names=datasets['target_names']))
    print(metrics.confusion_matrix(y_final_test, y_final_pred))

# Save the evaluation to a csv
# predictions_human_readable = np.column_stack((np.array(x_raw),
#                                               [int(prediction) for prediction in y_final_pred],
#                                               [ "{}".format(probability) for probability in all_probabilities]))
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
# print("Saving evaluation to {0}".format(out_path))
# with open(out_path, 'w') as f:
#     csv.writer(f).writerows(predictions_human_readable)

