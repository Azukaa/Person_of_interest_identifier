#!/usr/bin/python3
import csv
import pandas as pd
import numpy as np
import sys
import pickle
from sklearn.feature_selection import SelectKBest, f_classif
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier, load_classifier_and_data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### First analyze the dataset to select the features and use selectK best to identify the most important features
### Using pandas to do this

data = pd.read_csv("enron_dataset2.csv", index_col = 0 )

### Drop columns that have categorical data and columns that have more than 50% NaNs, we check these columns as below
#print(data.isnull().sum())

data.drop(["restricted_stock_deferred","loan_advances", "director_fees", "deferral_payments", "email_address"],axis = 1, inplace = True)

###Fill all NaN with 0
data.fillna(0, inplace = True)

### Using SelectKBest, select the top 10 features with the highest scores

X = data.drop("poi", axis = 1) # Features columnsx = x.astype(str)
y = data["poi"] # Label column
#print(X)
#print(y)
bestfeatures = SelectKBest(score_func = f_classif, k = "all")
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat the two dataframes for better visualization
features_scores=pd.concat([dfcolumns,dfscores], axis = 1)
features_scores.columns = ['features ', 'scores']
#print(features_scores.nlargest(5,'scores'))

### From the top 5 features gotten we add them to our features list

features_list = ['poi', 'bonus','exercised_stock_options','total_stock_value', 'salary','deferred_income'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset_unix.pkl", "rb") )


### Task 2: Remove outliers

data_dict.pop("TOTAL", 0)
data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("THE TRAVE AGENCY IN THE PARK",0)


### Task 3: Create new feature(s)
# For my new features, after noticing the variation between peoples salary and bonuses and also total stockvalue and total payments, I decided to combie these two features to create a new feature
# To do this, I computed the bonus to salary ratio for all individuals in the dataset

for person in data_dict:
    if data_dict[person]["bonus"] != "NaN" and data_dict[person]["salary"] != "NaN":
        bonus_salary_ratio = float(data_dict[person]["bonus"])/ float(data_dict[person]["salary"])
        data_dict[person]["bonus_salary_ratio"] = bonus_salary_ratio
    else:
        data_dict[person]["bonus_salary_ratio"] = "NaN"

for person in data_dict:
    if data_dict[person]["total_stock_value"] != "NaN" and data_dict[person]["total_payments"] != "NaN":
        total_amount = data_dict[person]["total_stock_value"] + data_dict[person]["total_payments"]
        data_dict[person]["total_amount"] = total_amount
    else:
        data_dict[person]["total_amount"] = "NaN"
newd = pd.DataFrame(data_dict)
newdata = newd.T
data = newdata.to_csv("enron_datasetNF.csv")

### Add the two new features created into our features_list list
features_list.append("total_amount")
features_list.append("bonus_salary_ratio")
#print(features_list)
### Store to my_dataset for easy export below.
my_dataset = data_dict
#print(my_dataset)
### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# After trying various classifiers using the function below, the best classifier was found to be the decision tree classifier

select_best = SelectKBest(f_classif, k=3)
features = select_best.fit_transform(features, labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
DT_classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 42)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!
### In this place, I have decided to create a function that will be called so that i can test through all classifiers using the best features. The function is located in
### the poi_id_main.py file. For simplicity i put in a separate file. After running the function on various classififcation algorithms and tuning it on various parameters,
### I found out that the best classifieris the Decision tree classifier. It was also tuned to various parameters.
### Also, there were three features that were used to achieve these results. These features include: exerceised_stock_options, total_stock_value and bonus.
parameters1 = {"criterion": ["gini", "entropy"],
              "min_samples_split":[2,3,4,5,6,7],
              "max_features": ["auto", "log2","sqrt",None]}
classifier = GridSearchCV(DT_classifier, parameters1)
classifier.fit(features_train,labels_train)
classifier = classifier.best_estimator_

test_classifier(classifier,my_dataset, features_list)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(classifier, my_dataset, features_list)

