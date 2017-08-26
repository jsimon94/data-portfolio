#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','from_poi_to_this_person_ratio', 
# 				'from_this_person_to_poi_ratio'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
data_dict.pop("LOCKHART EUGENE E")

### Task 3: Create new feature(s)
for name in data_dict:
    data_point = data_dict[name]
    
    if data_point['bonus']!= 'NaN' and data_point['salary'] != 'NaN' and data_point['salary'] != 0:
        
        bonus = float(data_point['bonus'])
        salary = float(data_point['salary'])
        data_point["bonus to salary ratio"] = bonus/salary
    else:
        data_point["bonus to salary ratio"] ='NaN'  
    
    if data_point['to_messages'] != 'NaN' and  data_point['from_messages'] !='NaN' and data_point['from_poi_to_this_person'] != 'NaN' and data_point['from_this_person_to_poi'] != 'NaN': 
        # from_poi ratio and to_poi ratio
        to_messages = float(data_point['to_messages']) 
        from_messages = float(data_point['from_messages']) 
        from_poi = float(data_point['from_poi_to_this_person']) 
        to_poi =float(data_point['from_this_person_to_poi'])

        data_point['from_poi_to_this_person_ratio'] = from_poi / to_messages
        data_point['from_this_person_to_poi_ratio'] = to_poi / from_messages
    
    else:
        data_point['from_poi_to_this_person_ratio'] = "NaN"
        data_point['from_this_person_to_poi_ratio'] = "NaN"
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
features_list = ['poi','bonus to salary ratio', 'salary', 'bonus',
                'from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio',
                 'from_poi_to_this_person','from_this_person_to_poi',
                 'deferral_payments', 'total_payments', 'loan_advances', 
                 'restricted_stock_deferred', 'deferred_income', 
                 'total_stock_value', 'expenses', 'exercised_stock_options', 
                 'long_term_incentive', 'restricted_stock', 'director_fees',
                  'shared_receipt_with_poi'] 
#The list above is the first set of features.                 
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#Split data into testing sets
from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(labels, 1000, test_size = 0.3, random_state = 42)
for train_index, test_index in sss:
    features_train, features_test = [features[i] for i in train_index], [features[i] for i in test_index]
    labels_train, labels_test = [labels[i] for i in train_index],[labels[i] for i in test_index]
#Used SelectKBest to pick top 5 features
from sklearn.feature_selection import SelectKBest, f_classif
X_new = SelectKBest(f_classif, k = 5)
X_new.fit_transform(features_train, labels_train)
# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in X_new.scores_ ]
# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  X_new.pvalues_ ]
# Get SelectKBest feature names, whose indices are stored in 'X_new.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in X_new.get_support(indices=True)]

# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

# Print
print ' '
print 'Selected Features, Scores, P-Values'
print features_selected_tuple
### Task 4: Try a varity of classifiers

my_dataset = data_dict
features_list = ['poi','salary', 'from_this_person_to_poi_ratio', 
                 'total_stock_value', 'exercised_stock_options', 'bonus']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Splitting the data using Stratified Shuffle Split
from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(labels, 1000, test_size = 0.3, random_state = 1)

for train_index, test_index in sss:
    #print("TRAIN:", train_index, "TEST:", test_index)
    features_train, features_test = [features[i] for i in train_index], [features[i] for i in test_index]
    labels_train, labels_test = [labels[i] for i in train_index],[labels[i] for i in test_index]
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#Tried Different Algorithms shown below. Each algorithm will print
#out the results from test_classifier function from tester.py

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf = 1, criterion= 'gini', 
                             max_depth=2, splitter ='random',random_state=42)
dtree.fit(features_train,labels_train)
score = dtree.score(features_test,labels_test)
pred= dtree.predict(features_test)
print "Decision Tree Results"
print test_classifier(dtree, my_dataset, features_list)

#KNearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
knn = KNeighborsClassifier(n_neighbors=3)
scaler = StandardScaler()
#features train and testing data are scaled with StandardScaler.
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)
#fit the scaled data under knn
knn.fit(features_train_scaled,labels_train)
pred = knn.predict(features_test_scaled)
print "K-Nearest Neighbors Results"
print test_classifier(knn, my_dataset, features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Final Algorithm: Naive Bayes
my_dataset = data_dict
features_list = ['poi','salary', 'from_this_person_to_poi_ratio', 
                 'total_stock_value', 'exercised_stock_options', 'bonus']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(labels, 1000, test_size = 0.3, random_state = 1)

for train_index, test_index in sss:
    #print("TRAIN:", train_index, "TEST:", test_index)
    features_train, features_test = [features[i] for i in train_index], [features[i] for i in test_index]
    labels_train, labels_test = [labels[i] for i in train_index],[labels[i] for i in test_index]

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(features_train,labels_train)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)