# -*- coding: utf-8 -*-
import numpy as np
import os



y_final_test = []
y_final_pred = []
misclassified_files = []

if FLAGS.use_weight_type == 'support':
    Normalize = 0.0
    for r in report_data:
        Normalize = Normalize + r['support']
else:
    Normalize = 1.0

proba_vote_method = False



class proba_vote_method:
	def run(self, y_final_pred_map, target_names):
	    print("vote using marginalization on cnn proba results ")
	    # P(document class) = Sum_j P(document class| document part j) x P(document part j)
	    for file, val in y_final_pred_map.items():
	        # fill the new y_test with observed class
	        y_final_test.append(int(val[0][2]))
	        # get predicted class
	        doc_proba = 1.0 / float(len(val))
	        cpt = np.array(val)
	        cpt = cpt[:,3:]
	        class_proba = cpt.sum(axis=0) * doc_proba

	        # get index with max values in P(class)
	        results = np.argmax(class_proba,axis=0)

	        # fill the new y_test with predicted class
	        y_final_pred.append(int(results))
	        # find misclassified files
	        if(results != int(val[0][2])):
	            subtable = [0] * len(target_names)
	            for i in range(len(class_proba)):
	                subtable[i] = str(round(class_proba[i], 2))
	            path, file = os.path.split(file)
	            path, categ = os.path.split(path)
	            file = categ + "/" + file
	            obs = target_names[int(val[0][2])]
	            pred = target_names[int(results)]
	            summary_row = [file,obs,pred] + subtable
	            misclassified_files.append(summary_row)

class count_vote_method():
	def run(self, y_final_pred_map, target_names, report_data, use_weight_type = 'no'):
		if use_weight_type == 'support':
	    	Normalize = 0.0
		    for r in report_data:
		        Normalize = Normalize + r['support']
		else:
		    Normalize = 1.0

	    for file, val in y_final_pred_map.items():
	        hist = {}
	        # fill the new y_test with observed class
	        y_final_test.append(int(val[0][2]))
	        # get predicted class, and fill an histogram
	        for v in val:
	            if v[1] not in hist:
	                hist[v[1]] = 0
	            hist[v[1]] = hist[v[1]] + 1

	        #print(hist)
	        # apply weight on prediction (not yet used, doesn t improve accuracy)
	        if FLAGS.use_weight:
	            weight = 1.0
	            for categnb, freq in hist.items():
	                idx = int(categnb)
	                categstr = target_names[idx]
	                for r in report_data:
	                    if categstr in r['class']:
	                        weight = r[FLAGS.use_weight_type]
	                hist[categnb] = freq*weight/Normalize
	        # get key with max values in hist 
	        vals = list(hist.values())
	        keys = list(hist.keys())
	        results = keys[vals.index(max(vals))]
	        # fill the new y_test with predicted class
	        y_final_pred.append(int(results))
	        # find misclassified files
	        if(results != int(val[0][2])):
	            subtable = [0] * len(target_names)
	            for i, label_count in hist.items():
	                subtable[int(i)] = str(round(label_count, 2))
	            path, file = os.path.split(file)
	            path, categ = os.path.split(path)
	            file = categ + "/" + file
	            obs = target_names[int(val[0][2])]
	            pred = target_names[int(results)]
	            summary_row = [file,obs,pred] + subtable
	            misclassified_files.append(summary_row)

	    return y_final_test, y_final_pred, misclassified_files





def vote_class(y_final_pred_map, target_names, vote_method_name = "count"):
	for file, val in y_final_pred_map.items():
		if(vote_method_name == "count"):
			vote_method = count_vote_method()


