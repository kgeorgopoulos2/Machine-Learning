
# coding: utf-8

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske
import tensorflow as tf

data = pd.read_csv('C:\\Users\\Kostas\\Desktop\\MSc Cloud Computing\\3rd Block - Weeks 13-16\\Machine Learning\\OK solution\\train.csv')

data.head()

data['Survived'].mean()

data.groupby('Pclass').mean()

class_sex_grouping = data.groupby(['Pclass','Sex']).mean()
class_sex_grouping

class_sex_grouping['Survived'].plot.bar()

group_by_age = pd.cut(data["Age"], np.arange(0, 90, 10))
age_grouping = data.groupby(group_by_age).mean()
age_grouping['Survived'].plot.bar()

data.count()

def get_combined_data():
    # reading train data
    train = pd.read_csv('C:/Users/Kostas/Desktop/MSc Cloud Computing/3rd Block - Weeks 13-16/Machine Learning/OK solution/train.csv')
    
    # reading test data
    test = pd.read_csv('C:/Users/Kostas/Desktop/MSc Cloud Computing/3rd Block - Weeks 13-16/Machine Learning/OK solution/test.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived',1,inplace=True)
    

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)
    
    return combined

combined = get_combined_data()

combined.shape

data = data.drop(['Age','Cabin','Embarked'], axis=1)

data["SibSp"] = data["SibSp"].fillna("NA")

data = data.dropna()

data.count()

def preprocess_titanic_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Pclass = le.fit_transform(processed_df.Pclass)
    processed_df = processed_df.drop(['Name','Ticket','SibSp'],axis=1)
    return processed_df

processed_df = preprocess_titanic_df(data)

X = processed_df.drop(['Survived'], axis=1).values
y = processed_df['Survived'].values

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf_dt = tree.DecisionTreeClassifier(max_depth=10)

clf_dt.fit (X_train, y_train)
clf_dt.score (X_test, y_test)

shuffle_validator = cross_validation.ShuffleSplit(len(X), n_iter=20, test_size=0.2, random_state=0)
def test_classifier(clf):
    scores = cross_validation.cross_val_score(clf, X, y, cv=shuffle_validator)
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))

test_classifier(clf_dt)

clf_rf = ske.RandomForestClassifier(n_estimators=50)
test_classifier(clf_rf)

clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
test_classifier(clf_gb)

eclf = ske.VotingClassifier([('dt', clf_dt), ('rf', clf_rf), ('gb', clf_gb)])
test_classifier(eclf)

ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = ax)
