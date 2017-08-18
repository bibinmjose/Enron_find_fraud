from time import time
import sys
import pickle
import json
import numpy as np
sys.path.append("../tools/")
from parameters import add_features, boolean_filter
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import StratifiedShuffleSplit

"""==================================================
# Parsing Data and splitting into features and labels
====================================================="""

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
				'salary','bonus','to_messages',
				'deferral_payments','total_payments','exercised_stock_options',
				'restricted_stock','shared_receipt_with_poi',
				'restricted_stock_deferred','total_stock_value',
				'expenses','loan_advances','from_messages','other',
				'from_this_person_to_poi','director_fees',
				'deferred_income','long_term_incentive',
				'email_address','from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
[data_dict.pop(key) for key in ["TOTAL","LOCKHART EUGENE E", "THE TRAVEL AGENCY IN THE PARK"]]

### Task 3: Create new feature(s)
### when new_features = True, additional features are added
new_features = False
# new_features = True

if new_features:
	features_list, my_dataset = add_features(features_list, data_dict)
else:
	my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

"""=======================
# Tuning the Classifier
=========================="""
# Name and No of Features
NAME_FEATURES = features_list[1:]
n_features = 2#len(NAME_FEATURES)

# And output .txt file to write all the settings
output_name = "NB_"
id_=output_name + "27"

# Creating f2 scorer (a weighted f1 scorer). when beta =1, f2 =  f1.
# lower value of beta (<1) favours precision.
f2_scorer = make_scorer(fbeta_score, beta=1)

# Parameters to optimize
params = [{
# 'selK__k': np.arange(6,n_features),
# 'selK__score_func' : [f_classif]#, mutual_info_classif]
# 'pca__n_components': np.arange(1,n_features)
}]

# Pipeline for building the classifier
pipe = Pipeline(steps  = [
('MinMaxscaler',MinMaxScaler()),
# ('StandardScaler',StandardScaler()),
('pca', PCA(n_components=n_features)),
# ('selK',SelectKBest()),
('classifNB', GaussianNB())
])

clf = GridSearchCV(pipe, param_grid=params, scoring=f2_scorer, n_jobs = -1)

# fitting the classifier after GridSearch
clf.fit(features,labels)
clf_best = clf.best_estimator_

# get the selected features list
selected_features = NAME_FEATURES
# selected_features = boolean_filter(NAME_FEATURES, clf_best.named_steps['selK'].get_support())

# print the best estimators and selected features to screen
print "\nBest Parameters\t:\n", clf_best
# print "\nSelected Features and k-Scores\t:",boolean_filter(zip(NAME_FEATURES,clf_best.named_steps['selK'].scores_), clf_best.named_steps['selK'].get_support())
print "\nSelected Features\t:",selected_features

# cross validation and obtaining scores
accuracy, precision, recall, f1, f2 = test_classifier(clf_best, my_dataset, features_list)


"""===================================
write the parameters to text file
======================================"""

if precision >0.3 and recall >0.27:
	text_file = open(output_name+".txt", "a")
	text_file.write("\nID_\t:\t{0}\n".format(id_))
	text_file.write("new_features\t:{3}\n\nn_features\t:{0}\n\nClassifier\t:\n{1}\n\nSelected Features\t:\n{2}".format(n_features,clf_best,selected_features, new_features))
	text_file.write("\n\naccuracy\t:{0}\nprecision\t:{1}\nrecall\t\t:{2}\nf_1\t\t\t:{3}\nf_2\t\t\t:{4}\n".format(accuracy, precision, recall, f1, f2))
	text_file.write("\n\n======================================================================\n\n\n")
	text_file.close()

# """====================================
# # Writing the parameters to dictionary
# ======================================="""
# clf_parameter = {}
# clf_parameter["id"] = id_
# clf_parameter["n_features"] = n_features
# # clf_parameter["clf_values"] = clf_values
# clf_parameter["steps"] = clf_best.named_steps.keys()
# clf_parameter["selected_features"] = selected_features
# clf_parameter["accuracy"] = accuracy
# clf_parameter["precision"] = precision
# clf_parameter["recall"] = recall
# clf_parameter["f1"] = f1
# clf_parameter["f2"] = f2

# print clf_parameter
"""====================================
Dump classifier and parameters to JSON
======================================="""
# dump_classifier_and_data(clf_best, my_dataset, selected_features)

# with open(output_name+'.json', 'a') as f:
#     json.dump(clf_parameter,f)
