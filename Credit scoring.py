# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 19:27:36 2017

@author: Erdi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_excel('Credit_Scoring_train.xls')
df_test = pd.read_excel('Credit_Scoring_test.xls')

print(df_train.head())

for i in df_train.columns:
    print(i)
    
print np.shape(df_train)

print df_train['LIMIT_BAL'].describe()
print df_train['AGE'].describe()
#print columns with remaining_amount in it
print df_train[[i for i in df_train.columns if 'REMAINING_AMOUNT' in i]].describe()
#print columns with payment in it]
print df_train[[i for i in df_train.columns if 'PAYMENT' in i]].describe()


#Remove the customer with empty 'LIMIT_BAL' value
df_train=df_train[df_train.LIMIT_BAL>0]

from sklearn.preprocessing import Imputer
#Create an imputer and replace NaN with median age
impute=Imputer(missing_values='NaN',strategy='median',axis=1) #axis=1 kolon

df_train['AGE']=impute.fit_transform(df_train['AGE'])

print(np.shape(df_train))
print(df_train['AGE'].describe())
print df_train['LIMIT_BAL'].describe()

#columns- not numbers
print df_train['SEX'].describe()
print df_train['EDUCATION'].describe()
print df_train['MARRIAGE'].describe()

#Replace empty values with the string 'MISSING_EDU and 'MISSING_MAR' from these two columns
df_train['EDUCATION'].fillna('MISSING_EDU',inplace=True)
df_train['MARRIAGE'].fillna('MISSING_MAR',inplace=True)

print(df_train['EDUCATION'].head())
print(df_train['MARRIAGE'].head())

print df_train['EDUCATION'].describe()
print df_train['MARRIAGE'].describe()

#create dummy variables--> for categorical variables
df_education = df_train[ 'EDUCATION' ]
#Transform using get_dummies
df_education=pd.get_dummies(df_education)

print(df_education.head())

df_marriage = df_train[ 'MARRIAGE' ]
#Transform using get_dummies
df_marriage=pd.get_dummies(df_marriage)


print(df_marriage.head())

#Drop columns 'CUSTOMER_ID', 'EDUCATION', 'MARRIAGE' and join with the two dataframes df_education and df_marriage
df_train.drop(['CUSTOMER_ID', 'EDUCATION', 'MARRIAGE'],axis=1,inplace=True)

df_train=pd.concat([df_train,df_education,df_marriage],axis=1)
print(df_train.head())

#Create columns Male and Female and drop the sex column
df_train['MALE']=(df_train['SEX']=='Male')*1
df_train['FEMALE']=(df_train['SEX']=='Female')*1
df_train.drop('SEX',axis=1,inplace=True)

print df_train.head()


df_train['LIMIT_PER_AGE'] = df_train['LIMIT_BAL']/df_train['AGE']
df_train['RELATIVE_REMAINING_AMOUNT'] = df_train['REMAINING_AMOUNT_1']/df_train['LIMIT_BAL']

#Create test data
df_test = pd.read_excel('Credit_Scoring_test.xls')
df_test['AGE'] = impute.transform(df_test['AGE']).transpose()

df_test['EDUCATION'].fillna('MISSING_EDU', inplace = True)
df_test['MARRIAGE'].fillna('MISSING_MAR', inplace = True)

df_education = df_test[ 'EDUCATION' ]
df_education = pd.get_dummies(df_education)

df_marriage = df_test[ 'MARRIAGE' ]
df_marriage = pd.get_dummies(df_marriage)

df_test.drop(['CUSTOMER_ID', 'EDUCATION', 'MARRIAGE'] , axis=1, inplace=True)

df_test = pd.concat([df_test, df_education, df_marriage], axis=1)

df_test['LIMIT_PER_AGE'] = df_test['LIMIT_BAL'] / df_test['AGE']
df_test['RELATIVE_REMAINING_AMOUNT'] = df_test['REMAINING_AMOUNT_1'] / df_test['LIMIT_BAL']

df_test['MALE'] = (df_test.SEX == 'Male')*1
df_test['FEMALE'] = (df_test.SEX == 'Female')*1
df_test.drop('SEX', axis=1, inplace=True)

df_test = df_test[df_train.columns]

##Classification and Evaluation######
#Set and drop the y variable (Column name 'default') for training and testing
y_train=df_train['default']
y_test=df_test['default']

df_train.drop('default',axis=1,inplace=True)
df_test.drop('default',axis=1,inplace=True)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#Train these 3 models with default parameters
#Train these 3 models with default parameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#Train these 3 models with default parameters
dt = DecisionTreeClassifier()
rf= RandomForestClassifier()
gbc = GradientBoostingClassifier()

dt.fit(df_train,y_train)
rf.fit(df_train,y_train)
gbc.fit(df_train,y_train)

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

print 'Decision Tree'
print 'Accuracy'
print(accuracy_score(y_test, dt.predict(df_test)))
print 'AUC score'
print(roc_auc_score(y_test, dt.predict_proba(df_test)[:,1]))
print 'Confusion Matrix'
print(confusion_matrix(y_test, dt.predict(df_test)))

print 'Random Forest'
print 'Accuracy'
print(accuracy_score(y_test, rf.predict(df_test)))
print 'AUC score'
print(roc_auc_score(y_test, rf.predict_proba(df_test)[:,1]))
print 'Confusion Matrix'
print(confusion_matrix(y_test, rf.predict(df_test)))

print 'Gradient Boosting'
print 'Accuracy'
print(accuracy_score(y_test, gbc.predict(df_test)))
print 'AUC score'
print(roc_auc_score(y_test, gbc.predict_proba(df_test)[:,1]))
print 'Confusion Matrix'
print(confusion_matrix(y_test, gbc.predict(df_test)))

#classifier function
def evaluate(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    print clf
    print 'Accuracy'
    print(accuracy_score(y_test, y_pred))
    print 'AUC score'
    print(roc_auc_score(y_test, y_pred_proba[:,1]))
    print 'Confusion Matrix'
    print(confusion_matrix(y_test, y_pred))


dt = DecisionTreeClassifier()
evaluate(dt, df_train, y_train, df_test, y_test)

#ROC curve comparison
a_dt , b_dt, c_dt = roc_curve(y_test, dt.predict_proba(df_test)[:,1])
a_rf , b_rf, c_rf = roc_curve(y_test, rf.predict_proba(df_test)[:,1])
a_gbc , b_gbc, c_gbc = roc_curve(y_test, gbc.predict_proba(df_test)[:,1])

plt.figure(figsize=(10,10))

plt.plot(a_dt,b_dt, c='red', label = 'Decision Tree', linewidth = 3)
plt.plot(a_rf,b_rf, c='blue', label = 'Random Forests', linewidth = 3)
plt.plot(a_gbc,b_gbc, c='black', label = 'Gradient Boosting', linewidth = 3)
plt.title('Area Under Curve', fontsize = 16)
plt.ylabel('True positive rate', fontsize = 14)
plt.xlabel('1 - True negative rate', fontsize = 14)

plt.legend(loc = 4)
plt.show()


#FINE TUNING
n_estimators = 1000

#Train GBC using 1000 estimators
gbc=GradientBoostingClassifier(n_estimators=n_estimators, verbose=1)
gbc.fit(df_train,y_train)

score = np.zeros(n_estimators)
for i, y_pred in enumerate(gbc.staged_predict_proba(df_test)):
    score[i] = roc_auc_score(y_test, y_pred[:,1])

score_train = np.zeros(n_estimators)
for i, y_pred in enumerate(gbc.staged_predict_proba(df_train)):
    score_train[i] = roc_auc_score(y_train, y_pred[:,1])
    
plt.figure(figsize=(10,5))
# Plot two different auc scores wrt the number of estimators
score = np.zeros(n_estimators)
for i, y_pred in enumerate(gbc.staged_predict_proba(df_test)):
    score[i] = roc_auc_score(y_test, y_pred[:,1])

score_train = np.zeros(n_estimators)
for i, y_pred in enumerate(gbc.staged_predict_proba(df_train)):
    score_train[i] = roc_auc_score(y_train, y_pred[:,1])
    
plt.figure(figsize=(10,5))
# Plot two different auc scores wrt the number of estimators
plt.plot(score, c='b',label='Testing')
plt.plot(score_train, c='r', label='Testing')
plt.legend()
plt.ylim((0.5,1))
plt.show()
plt.show()

#Gridsearch fine tuning

from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV
from time import time 
# specify parameters and distributions to sample from
parameters = {"max_depth": [3, 5, 10],
              "min_samples_leaf": [20, 50, 100],
              "n_estimators": [50, 100, 200]}

rf = RandomForestClassifier()
start = time()
#Carry a grid search with the given parameters. In total this will fit 3*3*3=27 models
grid_search=GridSearchCV(rf,parameters)
grid_search.fit(df_train,y_train)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))

print grid_search.best_estimator_
print grid_search.best_score_

print roc_auc_score(y_test, grid_search.best_estimator_.predict_proba(df_test)[:,1])

# Play with some parameters to see if it improves decision tree that we create

dt = DecisionTreeClassifier(criterion='entropy',max_depth=2,min_samples_leaf=50)
dt.fit(df_train,y_train)

print 'Decision Tree'
print 'Accuracy'
print(accuracy_score(y_test, dt.predict(df_test)))
print 'AUC score'
print(roc_auc_score(y_test, dt.predict_proba(df_test)[:,1]))
print 'Confusion Matrix'
print(confusion_matrix(y_test, dt.predict(df_test)))

from sklearn.tree import export_graphviz

export_graphviz(dt, out_file='Tree.dot', feature_names= df_train.columns)


###FEATURE IMPORTANCE
# Plot feature importance
feature_importance = gbc.feature_importances_

print(feature_importance)

df_features = pd.DataFrame(feature_importance)
df_features.columns = ['Importance']
df_features.index = df_train.columns
print(df_features)
# make importances relative to max importance and draw a barh plot
feature_importance=100*(feature_importance/feature_importance.max())
sorted_idx= np.argsort(feature_importance)
pos=np.arange(sorted_idx.shape[0])+ .5
plt.figure(figsize=(10,15))
plt.barh(pos,feature_importance[sorted_idx],align='center')
plt.yticks(pos,df_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

###Recursive Feature Elimination( Wrapper)
from sklearn.feature_selection import RFE

# Select 10 most important features by eliminating two features at each iteration
gclf=GradientBoostingClassifier()
selector=RFE(gclf,n_features_to_select=10,step=5)
selector.fit(df_train,y_train)
print(selector.support_)
print(df_train.columns[selector.support_])

#Create two dataframes containing the 10 most important features and evaluate
new_df_train= selector.transform(df_train)
new_df_test= selector.transform(df_test)
print np.shape(new_df_train)

evaluate(gclf,new_df_train,y_train,new_df_test,y_test) #evaluate is function defined before

###SCALER
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_train_scaled = scaler.fit_transform(df_train)

df_test_scaled = scaler.transform(df_test)
