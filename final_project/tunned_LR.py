from time import time
import sys
import pickle
import numpy as np
sys.path.append("../tools/")
from parameters import add_features
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

### Task 1: Select what features you'll use.
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
# You will need to use more features
new_features = ['bonus_log','salary_log','other_log','expenses_sqrt',
                'total_payments_log']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
[data_dict.pop(key) for key in ["TOTAL","LOCKHART EUGENE E", "THE TRAVEL AGENCY IN THE PARK"]]

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# Updating features list with newly defined features
features_list, my_dataset = add_features(features_list, data_dict)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

pipe = Pipeline(steps =[
	('StandardScaler', StandardScaler(copy=True, with_mean=True, with_std=True)), 
	('pca', PCA(copy=True, iterated_power='auto', n_components=1, random_state=None,
  						svd_solver='auto', tol=0.0, whiten=False)), 
	('classifLR', LogisticRegression(C=9.9999999999999995e-07, 
		class_weight=None, random_state=None,
          solver='liblinear', tol=1e-06, verbose=0, warm_start=False))])
# fitting the classifier after Grid search
pipe.fit(features,labels)

dump_classifier_and_data(pipe, my_dataset, features_list)
