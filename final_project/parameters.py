"""
This file contains various functions to generate parameters to be passed on to functions

"""

import numpy as np
from scipy.stats import rankdata

def add_features(features_list, data_dict):
	"""
	This function add new features to the incoming dict and returns a new dict with added features
	"""
	import numpy as np

	# Name of new features
	new_features = ['bonus_log','salary_log','other_log','expenses_sqrt','total_payments_log']

	for name in data_dict:
			if data_dict[name]['bonus'] != 'NaN':
				data_dict[name]['bonus_log'] = np.log(1.0+data_dict[name]['bonus'])
			else:
				data_dict[name]['bonus_log'] = "NaN"
			
			if data_dict[name]['salary'] != 'NaN':
				data_dict[name]['salary_log'] = np.log(1.0+data_dict[name]['salary']+1.0)
			else:
				data_dict[name]['salary_log'] = "NaN"
	
			if data_dict[name]['other'] != 'NaN':
				data_dict[name]['other_log'] = np.log(1.0+data_dict[name]['other'])
			else:
				data_dict[name]['other_log'] = "NaN"

			if data_dict[name]['expenses'] != 'NaN' and data_dict[name]['expenses']>0:
				data_dict[name]['expenses_sqrt'] = np.sqrt(data_dict[name]['expenses']+1.0)
			else:
				data_dict[name]['expenses_sqrt'] = "NaN"

			if data_dict[name]['total_payments'] != 'NaN' and data_dict[name]['total_payments'] >0 :
				data_dict[name]['total_payments_log'] = np.log(1.0+data_dict[name]['total_payments']+1.0)
			else:
				data_dict[name]['total_payments_log'] = "NaN"

	return features_list+new_features, data_dict

def boolean_filter(b_list, boolean):
	"""
	This function returns values in b_list where the boolean is true
	"""

	return [item for i, item in enumerate(b_list) if boolean[i]==True]

def top_n(list_array, n = 1):

	"""
	Returns a boolean mask with "True" for greatest "n" number of values
	"""

	np_array = np.array(list_array)
	# creating a mask
	mask = np.zeros(len(np_array.flatten()), dtype=bool)
	# rank matrix with highest value =1
	r =rankdata(np_array, method ="dense")
	r=(r.max()+1)-r

	for index, val in enumerate(r):
		if  val <= (n):
			mask[index] = True
	return mask.reshape(np_array.shape)





