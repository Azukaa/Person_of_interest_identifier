#!/usr/bin/python3
import csv
import pandas as pd
import numpy as np
import sys
import pickle
sys.path.append("../tools/")
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from operator import itemgetter
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### First analyze the dataset to select the features and use selectK best to identify the most important features
### Using pandas to do this

data = pd.read_csv("../final_project/enron_dataset2.csv", index_col = 0 )

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

features_list = ['poi', 'exercised_stock_options','total_stock_value','bonus', 'salary','deferred_income'] # You will need to use more features

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
data = newdata.to_csv("../final_project/enron_datasetNF.csv")

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



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
### In this place, I have decided to create a function that will be called so that i can test through all classifiers using the best features

def select_best_classifier(n_features, features, labels):
    clf_list = []
    select_best = SelectKBest(f_classif, k=n_features)
    features = select_best.fit_transform(features, labels)
    scores = select_best.scores_
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    #NB = GaussianNB()
    # dt = DecisionTreeClassifier()
    # parameters1 = {"criterion": ["gini", "entropy"],
    #               "min_samples_split":[2,3,4,5,6,7],
    #               "max_features": ["auto", "log2","sqrt",None]}
    rf = RandomForestClassifier()
    parameters2 = {'min_samples_split': [2, 3, 4, 5, 6, 7],
                  'max_features': ['auto', 'sqrt', 'log2', None],
                  'criterion': ['gini', 'entropy'],
                  'n_estimators': [2, 3, 4, 5, 6, 7]}
    # svm = SVC()
    # parameters3 = {'kernel': ['rbf'],
    #               'C': [1, 10, 100, 1000, 10000, 100000]}
    clf = GridSearchCV(rf, parameters2)
    clf.fit(features_train,labels_train)
    clf = clf.best_estimator_
    pred = clf.predict(features_test)
    accuracy = accuracy_score(pred, labels_test)
    precision = precision_score(pred, labels_test)
    recall = recall_score(pred, labels_test)

    count = 0
    for ele in labels_test:
        if ele == 1.0:
            count+=1
    new = zip(labels_test,pred)
    c = list(new)
    count1 = 0
    for elem in c:
        a, b = elem
        if a == 1.0 and b == 1.0:
            count1+=1
    clf_list.append([accuracy,precision,recall,n_features,scores,count,count1, clf])

    return clf_list[::-1][0]


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
clf_list2 = []
for num in range(1,len(features_list)):
    print(num,select_best_classifier(num,features, labels))
    clf_list2.append(select_best_classifier(num,features, labels))
print(clf_list2)
order_clf_list = sorted(clf_list2, key=itemgetter(1, 2))  # order by f1-score and accuracy
print(order_clf_list)
final = order_clf_list[len(order_clf_list) - 1][7]
print(final)

number_of_features = order_clf_list[len(order_clf_list) - 1][3]
print(number_of_features)

print('Features and scores: ')
score_list = order_clf_list[len(order_clf_list) - 1][4]
print(score_list)
features = features_list[1:]
features_scores = []
i = 0
for feature in features:
    features_scores.append([feature, score_list[i]])
    i += 1
features_scores = sorted(features_scores, key=itemgetter(1))
print(features_scores[::-1])

print ('Features used: ')
new_features_list = []
for feature in features_scores[::-1][:number_of_features]:
    new_features_list.append(feature[0])
print(new_features_list)
new_features_list = ['poi'] + new_features_list
print(new_features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
