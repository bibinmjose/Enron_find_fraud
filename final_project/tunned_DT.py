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
from sklearn.tree import export_graphviz
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

t0=time()

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
# new_features = False
features_list, my_dataset = add_features(features_list, data_dict)


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

"""=======================
# Tuning the Classifier
=========================="""

# And output .txt file to write all the settings
output_name = "DT_"
id_=output_name + "tunned"

# Creating f2 scorer (a weighted f1 scorer). when beta =1, f2 =  f1.
# lower value of beta (<1) favours precision.
f2_scorer = make_scorer(fbeta_score, beta=1)

# Parameters to optimize
params = [{
}]


# Pipeline for building the classifier
pipe = Pipeline(steps=[('MinMaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('classifDT', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features='log2', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=True, random_state=42, splitter='best'))])


# fitting the classifier after GridSearch
clf_best = pipe.fit(features,labels)
t1=time()
print "\nTime 1\t:",time()-t0

# dump classifier and features
dump_classifier_and_data(clf_best, my_dataset, features_list)

# cross validation and obtaining scores
accuracy, precision, recall, f1, f2 = test_classifier(clf_best, my_dataset, features_list)


print "\nTime 2\t:",time()-t1

	
# """============================================================
# # Visualize decision tree and converting into the pdf file
# ==============================================================="""

import pydotplus

export_graphviz(decision_tree = clf_best.named_steps['classifDT'], out_file =id_+'.dot',
	feature_names=features_list[1:], class_names="no",filled=True, rounded=True,
	special_characters=True)

pydotplus.graph_from_dot_file(id_+'.dot').write_pdf(id_+'.pdf')
