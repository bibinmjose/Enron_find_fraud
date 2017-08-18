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
from sklearn.tree import ExtraTreeClassifier

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
new_features = True

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
n_features = int(len(NAME_FEATURES))

# And output .txt file to write all the settings
output_name = "ET_"
id_=output_name + "5"

# Creating f2 scorer (a weighted f1 scorer). when beta =1, f2 =  f1.
# lower value of beta (<1) favours precision.
f2_scorer = make_scorer(fbeta_score, beta=5)

# Parameters to optimize
params = [{
# 'classifADB__n_estimators': [10,50,100],
# 'classifADB__learning_rate': [.1,.5,1.0],
# 'classifET__criterion':["gini","entropy"],
'classifET__max_features':["sqrt","log2"],
'classifET__max_depth':np.arange(3,10),
'classifET__random_state':[42,43,45],
'classifET__max_leaf_nodes':np.arange(2,10),
'classifET__min_samples_leaf':np.arange(2,10),
# 'selK__k': [6,7,8,9],#np.arange(1,n_features),
# 'selK__score_func' : [f_classif,mutual_info_classif]
# 'pca__n_components': np.arange(1,n_features),
# 'rfe__n_features_to_select': [6],#np.arange(1,n_features),
}]


# Pipeline for building the classifier
pipe = Pipeline(steps  = [
# ('MinMaxscaler',MinMaxScaler()),
# ('StandardScaler',StandardScaler()),
# ('pca', PCA()),
# ('selK',SelectKBest()),
# ('rfe',RFE(DecisionTreeClassifier(), step=1)),
('classifET', ExtraTreeClassifier(random_state = 42))
])

clf = GridSearchCV(pipe, param_grid=params, scoring=f2_scorer,cv=10, n_jobs = -1)

# fitting the classifier after GridSearch
clf.fit(features,labels)
clf_best = clf.best_estimator_

# get the selected features list
selected_features = NAME_FEATURES
# selected_features = boolean_filter(NAME_FEATURES, clf_best.named_steps['selK'].get_support())
# selected_features = boolean_filter(NAME_FEATURES, clf_best.named_steps['rfe'].get_support())

t1=time()
print "\nTime 1\t:",time()-t0

# Print classifier and Selected features
print "\n",clf_best,"\n"
print "\nSelected Features\t:",selected_features,"\n"

# cross validation and obtaining scores
accuracy, precision, recall, f1, f2 = test_classifier(clf_best, my_dataset, features_list)


print "\nTime 2\t:",time()-t1

"""===================================
write the parameters to text file
======================================"""

if f1 >0.2:
	text_file = open(output_name+".txt", "a")
	text_file.write("\nID_\t:\t{0}\n".format(id_))
	text_file.write("new_features\t:{3}\n\nn_features\t:{0}\n\nClassifier\t:\n{1}\n\nSelected Features\t:\n{2}".format(n_features,clf_best,selected_features, new_features))
	text_file.write("\n\naccuracy\t:{0}\nprecision\t:{1}\nrecall\t\t:{2}\nf_1\t\t:{3}\nf_2\t\t:{4}\n".format(accuracy, precision, recall, f1, f2))
	text_file.write("\n\n======================================================================\n\n\n")
	text_file.close()

	
# """============================================================
# # Visualize decision tree and converting into the pdf file
# ==============================================================="""

	import pydotplus

	export_graphviz(decision_tree = clf_best.named_steps['classifET'], out_file =id_+'.dot',
		feature_names=NAME_FEATURES, class_names="no",filled=True, rounded=True,
		special_characters=True)

	pydotplus.graph_from_dot_file(id_+'.dot').write_pdf(id_+'.pdf')

