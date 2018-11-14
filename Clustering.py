print(__doc__)

import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GMM
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
import pandas as pd 
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
sns.set_style("whitegrid")

#Read and summarize the dataframe
df_train= pd.read_csv('Transactions_2015.csv')
print np.shape(df_train)

print df_train.head()
print df_train.describe()

#fillna-> fulfill NaN values by adding 0
df_train.fillna(0,inplace=True)

print(df_train.head())
print(df_train.describe())

#negative amount with respect to customer based on columns(count)
print df_train[df_train<0].count(axis=0) #count based on columns


print df_train[df_train[df_train<0].count(axis=1) >=1] #axis 1 is running for rows #<0 lari sayiyor

print np.shape(df_train[df_train[df_train<0].count(axis=1)<=0]) # True-False boolean matrix- satirlardaki truelari sayiyor
print np.shape(df_train[df_train[df_train<0].count(axis=1)<1])

#Create a dataframe df_sum consisting of total spendings of customers
print df_train.head()
df_train1=df_train[[i for i in df_train.columns if 'Category' in i]]
#df_train1= df_train.iloc[:,1:7]
df_sum=df_train1.sum(axis=1) #based on rows-meaning customers
print(df_sum.head())
print(df_sum.describe())

#create boxplot
plt.figure(figsize=(5,10))
plt.boxplot(x=df_sum.values, sym='ko')
plt.show()

#Print min, max and some percentile values
print df_sum.min()

print np.percentile(df_sum,25)
print np.percentile(df_sum,50)
print np.percentile(df_sum,95)
print np.percentile(df_sum,99)
print df_sum.max()

#Create a column called 'sum'
df_train['sum']=df_sum.values
print df_train.head()

df_train=df_train[df_train['sum']]<= np.percentile(df_train['sum'],99)

# draw correlation map
col_cat=['Category_1','Category_2','Category_3','Category_4','Category_5','Category_6']

df_train_cat=df_train[col_cat]

sns.heatmap(df_train_cat.corr())

df_train_cat.columns = ['Et-Balik', 'Kahvaltilik', 'Gida-Sekerleme', 'Sebze', 'Meyve', 'Deterjan']

#Join meyve and sebze columns and drop the individual columns
df_train_cat['Meyve-Sebze']=df_train_cat['Meyve']+df_train_cat['Sebze']

df_train_cat.drop(['Sebze','Meyve'],axis=1,inplace=True)
#Draw the correlation map again

sns.heatmap(df_train_cat.corr())
plt.show()

#CLUSTERING
kmeans=KMeans(n_clusters=5, n_init=10, random_state=0)
kmeans.fit(df_train_cat)

df_centers=pd.DataFrame(kmeans.cluster_centers_) #cluster centerlar
df_centers.columns=df_train_cat.columns

print(df_centers) 

#Draw a line plot (using plot function) of the cluster centers
plt.figure(figsize=(10,10))
plt.plot(df_centers.transpose().values)
plt.xticks(np.arange(5),df_train_cat.columns)
plt.show()

#Apply the necessary conversion / percentage 
print df_train_cat
df_train_cat_pct=df_train_cat.div(df_train_cat.sum(axis=1),axis=0)

kmeans=KMeans(n_clusters=5, n_init=10, random_state=0)
kmeans.fit(df_train_cat_pct)
df_centers=pd.DataFrame(kmeans.cluster_centers_) #cluster centerlar
df_centers.columns=df_train_cat_pct.columns

print(df_centers) 

#Draw a line plot (using plot function) of the cluster centers
plt.figure(figsize=(10,10))
plt.plot(df_centers.transpose().values)
plt.xticks(np.arange(5),df_train_cat.columns)
plt.show()

#How many customers belong to each segment? Use np.unique
a,b=np.unique(kmeans.labels_,return_counts=True)
print a
print b

#five cluster
x_ticks = df_train_cat.columns
ncols = 5 
x_range = range(0, ncols)
df0 = df_train_cat[kmeans.labels_ == 0]
df1 = df_train_cat[kmeans.labels_ == 1]
df2 = df_train_cat[kmeans.labels_ == 2]
df3 = df_train_cat[kmeans.labels_ == 3]
df4 = df_train_cat[kmeans.labels_ == 4]

fig = plt.figure(figsize=(10, 10))
df0 = np.asarray(df0.transpose())
df1 = np.asarray(df1.transpose())
df2 = np.asarray(df2.transpose())
df3 = np.asarray(df3.transpose())
df4 = np.asarray(df4.transpose())
plt.plot(df0, color="blue", alpha=0.1)
plt.plot(df1, color="red", alpha=0.1)
plt.plot(df2, color="green", alpha=0.1)
plt.plot(df3, color="orange", alpha=0.1)
plt.plot(df4, color="purple", alpha=0.1)

x_range = range(0, ncols)
plt.xlabel("Product Categories", fontsize = 14)
plt.ylabel("Percentage of Shopping", fontsize = 14)

plt.xticks(x_range, x_ticks)
plt.title('Parallel Line Plot for Customer Segmentation', fontsize=16)
plt.show()

# five clusters based on percentages
x_ticks = df_train_cat.columns
ncols = 5 
x_range = range(0, ncols)
df0 = df_train_cat_pct[kmeans.labels_ == 0]
df1 = df_train_cat_pct[kmeans.labels_ == 1]
df2 = df_train_cat_pct[kmeans.labels_ == 2]
df3 = df_train_cat_pct[kmeans.labels_ == 3]
df4 = df_train_cat_pct[kmeans.labels_ == 4]

fig = plt.figure(figsize=(10, 10))
df0 = np.asarray(df0.transpose())
df1 = np.asarray(df1.transpose())
df2 = np.asarray(df2.transpose())
df3 = np.asarray(df3.transpose())
df4 = np.asarray(df4.transpose())
plt.plot(df0, color="blue", alpha=0.1)
plt.plot(df1, color="red", alpha=0.1)
plt.plot(df2, color="green", alpha=0.1)
plt.plot(df3, color="orange", alpha=0.1)
plt.plot(df4, color="purple", alpha=0.1)

x_range = range(0, ncols)
plt.xlabel("Product Categories", fontsize = 14)
plt.ylabel("Percentage of Shopping", fontsize = 14)

plt.xticks(x_range, x_ticks)
plt.title('Parallel Line Plot for Customer Segmentation', fontsize=16)
plt.show()

#After obtaining customer segments, we can try and see which segments are more important, for that we use pie chart and compare the number of customers in a cluster and the total spending of the same customers
df_train_cat['label'] = kmeans.predict(df_train_cat_pct)
df = pd.DataFrame(df_train_cat['label'])
df['sum'] = df_train_cat.sum(axis=1)

#Plot a pie chart of total spendings per label
print (df.head())
var=df.groupby(['label']).sum().stack()
temp=var.unstack()
print temp
x_list=temp['sum']
label_list=temp.index
plt.axis("equal")
plt.pie(x_list,labels=label_list,autopct="%1.1f%%")
plt.title("Spending Percentages")
plt.show()
print x_list

#Plot a pie chart of number of customers per label
labels,label_counts=np.unique(df['label'].values,return_counts=True)
print label_counts

plt.axis("equal")
plt.pie(label_counts,labels=labels,autopct="%1.1f%%")
plt.title("Number of Customers")
plt.show()

