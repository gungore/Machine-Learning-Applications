# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 19:00:53 2017

@author: Erdig
"""
from pymining import seqmining
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import operator

#Instacart data is imported to Python.
order_products_train_df = pd.read_csv("order_products__train.csv") 
order_products_prior_df = pd.read_csv("order_products__prior.csv") 
orders_df = pd.read_csv("orders.csv") 
products_df = pd.read_csv("products.csv") 
aisles_df = pd.read_csv("aisles.csv") 
departments_df = pd.read_csv("departments.csv")

#Analysis to see whether the count of orders are changed wrt a hour in a day
fig,ax = plt.subplots() 
fig.set_size_inches(20,5) 
sn.countplot(data=orders_df,x="order_hour_of_day",ax=ax,color="#34495e") 
ax.set(xlabel='Hour Of The Day',title="Order Count Across Hour Of The Day")

#The day is divided 3 group to see differentiation more.
orders_df['TimeGroup'] = 3
orders_df['TimeGroup'][(orders_df['order_hour_of_day'] >= 1) & (orders_df['order_hour_of_day'] <= 8) ] = 1
orders_df['TimeGroup'][(orders_df['order_hour_of_day'] >= 9) & (orders_df['order_hour_of_day'] <= 16)] = 2


#Analysis to see whether the count of orders are changed wrt a time groups ia a day
fig,ax = plt.subplots() 
fig.set_size_inches(20,5) 
sn.countplot(data=orders_df,x="TimeGroup",ax=ax,color="#34495e") 
ax.set(xlabel='Hour Of The Day',title="Order Count Across Hour Of The Day")

def drawn_graph(dataframe_items):
    grouped = items.groupby(["TimeGroup", "department"])["product_id"].aggregate({'Total_products': 'count'}).reset_index()
    grouped.sort_values(by='Total_products', ascending=False, inplace=True)
    
    fig, axes = plt.subplots(7,3, figsize=(20,45), gridspec_kw =  dict(hspace=1.4))
    for (department, group), ax in zip(grouped.groupby(["TimeGroup"]), axes.flatten()):
        g = sn.barplot(group.department, group.Total_products , ax=ax)
        ax.set(xlabel = "department", ylabel=" Number of products")
        g.set_xticklabels(labels = group.department,rotation=90, fontsize=12)
        ax.set_title(department, fontsize=15)
    plt.show()
    
    
    
    plt.figure(figsize=(12,8)) 
    sn.countplot(x="order_dow", data=orders_df_prior) 
    plt.ylabel('Count', fontsize=12) 
    plt.xlabel('Day of week', fontsize=12) 
    plt.xticks(rotation='vertical')
    plt.title("Frequency of order by week day", fontsize=15) 
    plt.show()

#Prior order data is prepared. And other data sources are joinned to prior data-> items dataframe
orders_df_prior=orders_df[orders_df['eval_set']=='prior']
items  =pd.merge(left= pd.merge(left =pd.merge(left=orders_df_prior, right=order_products_prior_df, how='left'), right=products_df, how='left'),right=departments_df,how='left')

print (items.head())
  

#Input Variables are created for supervised learning
variable1_df=items.groupby("user_id",as_index=False).agg(
        {"order_id": ["count", "nunique"],  #nunique-->total number of orders
         "product_id": "nunique",
         "reordered":["sum", "mean"], #total reordered
"department_id":"nunique","aisle_id":"nunique",
"order_dow":"nunique"
    })


variable2_df=items.groupby(['user_id','product_id'],as_index=False).agg(
        {
         "add_to_cart_order": "mean",
         "reordered":"sum",  #productbesed reordered
"order_hour_of_day":["max","mean"],
"order_number":["min","max"]
    })



variable3_df=items.groupby(['user_id','product_id','order_hour_of_day'],as_index=False).agg({
         "order_id": "count"})


variable4_df= variable3_df.groupby(['user_id','product_id'],as_index=False).agg(
        {
   "order_id":"max"
    })
 

variable5_df  =pd.merge(left=variable3_df, right=variable4_df, how='left',on = ['user_id','product_id'])

 
variable5_df=variable5_df[variable5_df['order_id_x']==variable5_df['order_id_y']]
variable5_df=variable5_df.groupby(['user_id','product_id']).first().reset_index()
del variable5_df['order_id_x']
del variable5_df['order_id_y']

orders_df1=orders_df[orders_df['eval_set']=='prior']
variable6_df= orders_df1.groupby(['user_id'],as_index=False).agg(
        {
   "days_since_prior_order":["sum","mean"]
    })


##TRAIN SET is generated from dataframes. Other columns are joinned to data. 
orders_train_df=orders_df[orders_df['eval_set']=='train']
items_train=pd.merge(left= pd.merge(left =pd.merge(left=orders_train_df, right=order_products_train_df, how='left'), right=products_df, how='left'),right=departments_df,how='left')
#Because we are trying to predict whether the product will be reordered or not, the prior information is also taken the base data.
items_appended=items.append(items_train)
items_appended2=items_appended[['user_id','order_number','product_id','reordered']]
#One product could be exist more than 1 in one user, we take the updated reordered information of user_id-product_id group
#If one product is reordered more than one the data contains as the same count to learn more this product.
items_appended3=items_appended2.groupby(['user_id','product_id'],as_index=False).agg({"reordered":"max"})

#Input variables are joinned to train data
items_train_m1=pd.merge(left=items_appended3,right=variable1_df,how='left',on=['user_id'])
items_train_m2=pd.merge(left=items_train_m1,right=variable2_df,how='left',on=['user_id','product_id'])
items_train_m3=pd.merge(left=items_train_m2,right=variable5_df,how='left',on=['user_id','product_id'])
items_train_m4=pd.merge(left=items_train_m3,right=variable6_df,how='left',on=['user_id'])
kmeans_output=kmeans()
items_train_m5=pd.merge(left=items_train_m4,right=kmeans_output,how='left',on=['user_id'])
 
items_train_m5=items_train_m5.drop('product_id_x',axis=1)
items_train_m5=items_train_m5.drop('user_id_x',axis=1)
items_train_m5['product_reorederd_rate']=items_train_m5[('reordered_y', 'sum')]/items_train_m5[('order_id','nunique')]
items_train_m5['order_since_last_order']=items_train_m5[('order_id','nunique')]-items_train_m5[('order_number_y','max')]
items_train_m5.columns
items_train_m5.head(500)
#create csv file of train data for R xgboost algorithm
items_train_m5.to_csv('items_train_setv1newfeatureskmeans.csv', sep=',',index=False) 
 

##TEST SET is generated from dataframes. Other columns are joinned to data. 
orders_test_df=orders_df[orders_df['eval_set']=='test']
#For test data prior products of user is joinned to test data. Because the possible reordered products should be joinned to test users.
items_test=pd.merge(left=orders_test_df,right=items,how='left',on=['user_id'] )
items_test.head(50)
items_test=items_test.rename(columns={'order_id_x': 'order_id','user_id_x':'user_id','eval_set_x':'eval_set','order_number_x':'order_number','order_dow_x':'order_dow','order_hour_of_day_x':'order_hour_of_day','days_since_prior_order_x':'days_since_prior_order'})
items_test=items_test[['order_id','user_id','order_number','product_id','reordered']]
#We take the updated reordered information of user_id-product_id group.It is aimed to take one record per user_id-product_id
items_test=items_test.sort_values('reordered',ascending=False).groupby(['user_id','product_id'],as_index=False).first().reset_index()
#items_test=items_test.groupby(['user_id','product_id'],as_index=False).agg({"reordered":"max"})

#Input variables are joinned to test data
items_test_m1=pd.merge(left=items_test,right=variable1_df,how='left',on=['user_id'])
items_test_m2=pd.merge(left=items_test_m1,right=variable2_df,how='left',on=['user_id','product_id'])
items_test_m3=pd.merge(left=items_test_m2,right=variable5_df,how='left',on=['user_id','product_id'])
items_test_m4=pd.merge(left=items_test_m3,right=variable6_df,how='left',on=['user_id'])
items_test_m4=pd.merge(left=items_test_m4,right=kmeans_output,how='left',on=['user_id'])
items_test_m4.columns 
items_test_m4.head()

#Order_id is necessary for xgboost output
order1=orders_test_df[['order_id','user_id']]
items_test_m4=pd.merge(items_test_m4,order1,how='inner',on='user_id') 
items_test_m4=items_test_m4.drop('product_id_x',axis=1)
items_test_m4=items_test_m4.drop('user_id_x',axis=1)
items_test_m4=items_test_m4.drop('index',axis=1)
items_test_m4=items_test_m4.drop('order_id_y',axis=1)
items_test_m4=items_test_m4.drop('order_number_x',axis=1)
items_test_m4=items_test_m4.rename(columns={'order_id_x': 'order_id'})
items_test_m4['product_reorederd_rate']=items_test_m4[('reordered_y', 'sum')]/items_test_m4[('order_id','nunique')]
items_test_m4['order_since_last_order']=items_test_m4[('order_id','nunique')]-items_test_m4[('order_number_y','max')]


#create csv file of test data for R xgboost algorithm
items_test_m4.to_csv('items_test_setv1newfeaturekmeans.csv', sep=',',index=False)


def kmeans():
    
    grouped = items.groupby(["user_id", "aisle_id"])["aisle_id"].aggregate({'Total_products': 'count'}).reset_index()
    grouped.sort_values(by='Total_products', ascending=False, inplace=True)    
    grouped_pivot=pd.pivot_table(grouped,'Total_products',index='user_id',columns='aisle_id')   
    grouped_2= items.groupby(["user_id"])["aisle_id"].aggregate({'Total_products': 'count'}).reset_index()    
    grouped_f=pd.merge(grouped_pivot,grouped_2,  how='left', left_index=True, right_on = ['user_id'])
    grouped_f.fillna(0,inplace=True)
    grouped_f2 = grouped_f.div(grouped_f.Total_products, axis='index')
    del grouped_f2['user_id']
   
    
    import pylab
    kmeans = KMeans(n_clusters = 5, n_init=10, random_state=0)
    kmeans.fit(grouped_f2)
    label_p=kmeans.predict(grouped_f2)
     
    df_centers_p = pd.DataFrame(kmeans.cluster_centers_)
    df_centers_p.columns = grouped_f2.columns
    dflabel_p=pd.DataFrame(label_p)
    dflabel_p.index = range(1,len(dflabel_p)+1)
    dflabel_p['user_id']=dflabel_p.index
    dflabel_p.columns=['Labels','user_id']
    
    
    del grouped
    del grouped_pivot
    del grouped_2
    del grouped_f
    del grouped_f2
    return dflabel_p

#Alternative Classification Algorithms
#read dataframe from csv due to memory error
items_train_m5= pd.read_csv("items_train_setv1newfeatureskmeans.csv") 
items_test_m4= pd.read_csv("items_test_setv1newfeaturekmeans.csv") 

train_label=items_train_m5['reordered_x']    
train_data=items_train_m5.drop('reordered_x',axis=1)
test_data=items_test_m4.drop('reordered_x',axis=1)
test_data.fillna(0,inplace=True)
train_data.fillna(0,inplace=True)
train_label.fillna(0,inplace=True)
test_data2=test_data.drop('order_id',axis=1)
test_data2=test_data2.drop('user_id',axis=1)
test_data2=test_data2.drop('product_id',axis=1)
train_data2=train_data.drop('user_id',axis=1)
train_data2=train_data2.drop('product_id',axis=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
gbc = GradientBoostingClassifier()
logistic = LogisticRegression()

##logistic Regression/Cross-validation
X_train, X_test, y_train, y_test = train_test_split(train_data2, train_label, test_size=0.33, random_state=0)
logistic.fit(X_train,y_train)
logistic.score(X_test,y_test)

#logistic Regression Prediction on Test data
logistic.fit(train_data2,train_label)
y_pred=logistic.predict(test_data2)

y_pred2=y_pred.reshape(len(y_pred),1)
test_data_result=test_data[['order_id','product_id']]
test_data_result['reordered']=y_pred2

test_data_result.to_csv('logistic_regressionresult.csv', sep=',',index=False)


##Decision Tree/Cross-validation 
X_train, X_test, y_train, y_test = train_test_split(train_data2, train_label, test_size=0.33, random_state=0)
dt.fit(X_train,y_train)
dt.score(X_test,y_test)

#Decision Tree Prediction on Test data
dt.fit(train_data2,train_label)
y_pred=dt.predict(test_data2)

y_pred2=y_pred.reshape(len(y_pred),1)
test_data_result=test_data[['order_id','product_id']]
test_data_result['reordered']=y_pred2

test_data_result.to_csv('decision_treeresult.csv', sep=',',index=False)

##Gradient Boosting/Cross-validation 
X_train, X_test, y_train, y_test = train_test_split(train_data2, train_label, test_size=0.33, random_state=0)
gbc.fit(X_train,y_train)
gbc.score(X_test,y_test)

#Gradient Boosting Prediction on Test data
gbc.fit(train_data2,train_label)
y_pred=gbc.predict(test_data2)

y_pred2=y_pred.reshape(len(y_pred),1)
test_data_result=test_data[['order_id','product_id']]
test_data_result['reordered']=y_pred2

test_data_result.to_csv('gbc_treeresult.csv', sep=',',index=False)

#Random Forest/Cross-validation 
X_train, X_test, y_train, y_test = train_test_split(train_data2, train_label, test_size=0.33, random_state=0)
rf.fit(X_train,y_train)
rf.score(X_test,y_test)


#Random Forest Prediction on Test data
rf.fit(train_data2,train_label)
y_pred=rf.predict(test_data2)

y_pred2=y_pred.reshape(len(y_pred),1)
test_data_result=test_data[['order_id','product_id']]
test_data_result['reordered']=y_pred2

test_data_result.to_csv('rf_result.csv', sep=',',index=False)



