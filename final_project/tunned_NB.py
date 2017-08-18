from time import time
import sys
import pickle
import numpy as np
sys.path.append("../tools/")
from parameters import add_features
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

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
### Store to my_dataset for easy export below.
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


# Defining pipeline
pipe = Pipeline(steps =[
	('selK', SelectKBest(k=8, score_func=f_classif)), 
	('classifNB', GaussianNB(priors=None))
	])

t0=time()
# fitting the classifier after Grid search
pipe.fit(features,labels)
print "Average time for learning\t:{0:5.2f}".format(time()-t0)
print pipe

dump_classifier_and_data(pipe, my_dataset, features_list)