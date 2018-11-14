#clustering mixture model
import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.normal(1000,200,(3000,1))
x2 = np.random.normal(2000,200,(3000,1))
x = np.concatenate((x1,x2))

plt.figure(figsize=(10,5))
plt.hist(x1,bins=25,color= 'blue')
plt.show()
plt.figure(figsize=(10,5))
plt.hist(x2,bins=25, color='red')
plt.show()

plt.figure(figsize=(10,5))
plt.hist(x, bins = 50)
plt.show()

from sklearn.mixture import GaussianMixture

gmm=GaussianMixture(n_components=2,random_state=0)
gmm.fit(x)

#Fit a GMM model with 2 components

x_new = np.arange(0,3001,1)
x_new = x_new.reshape((3001,1))

#Find the probability of belonging to component number 2
pred = gmm.predict_proba(x_new)[:,1]

plt.figure(figsize=(10,5))
plt.hist(x, bins = 50)
plt.xlim((0,3000))
plt.ylim((-50,450))
plt.plot(pred*400)
plt.show()

#We can also draw probabilities for both components to better illustrate the idea of mixture. An observation with a value 1000 belongs to the first mixture with probability 1 (almost). As the values of observations increase, the probability of belonging to the first components decreases (orange line, around 1250) and the probability of belonging to the second component increases (green line).
pred = gmm.predict_proba(x_new)

plt.figure(figsize=(10,5))
plt.hist(x, bins = 50)
plt.xlim((0,3000))
plt.ylim((-50,450))
plt.plot(pred*400)
plt.show()

#As you can see if a number is below 1500, then it is more likely to belong to the component with mean 1000. After that number, a number is more likely to belong to component with mean 2000, as we would expect, since the distributions only differ in their means.
#We can also carry the same analysis with a third distribution (with mean 3000) and a third component.
x3 = np.random.normal(3000,200,(3000,1))
x = np.concatenate((x1,x2,x3))

plt.figure(figsize=(10,5))
plt.hist(x, bins = 75)
plt.show()

gmm = GaussianMixture(n_components=3, random_state = 0)
gmm.fit(x)

print('Means for components')
print(gmm.means_)

x_new = np.arange(0,4000,1)
x_new = x_new.reshape((4000,1))

pred = gmm.predict_proba(x_new)


plt.figure(figsize=(10,5))
plt.hist(x, bins = 75)
plt.xlim((0,4000))
plt.ylim((-50,450))
plt.plot(pred*400)
plt.show()

#Customer Segmentation
print(__doc__)
#%matplotlib inline
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
import pandas as pd 
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
sns.set_style("whitegrid")

df_train = pd.read_csv('Transactions_2015.csv')

df_train.fillna(0,inplace = True)

negative_cells = df_train<0

df_train = df_train[negative_cells.sum(axis=1) < 1]

df_sum = df_train[['Category_' + str(i) for i in [1,2,3,4,5,6]]].sum(axis=1)

df_train['sum'] = df_sum

df_train = df_train[df_train['sum'] <= np.percentile(df_train['sum'], 99)]

col_cat = [ u'Category_1', u'Category_2', u'Category_3', u'Category_4', u'Category_5', u'Category_6']
#col_mon = [ u'last_amo_6', u'last_amo_5', u'last_amo_4', u'last_amo_3', u'last_amo_2', u'last_amo_1']

df_train_cat = df_train[col_cat]
#df_train_mon = df_train[col_mon]

df_train_cat.columns = ['Et-Balik', 'Kahvaltilik', 'Gida-Sekerleme', 'Sebze', 'Meyve', 'Deterjan']

df_train_cat['Sebze-Meyve'] = df_train_cat['Sebze'] + df_train_cat['Meyve']

df_train_cat.drop(['Sebze', 'Meyve'], axis=1, inplace=True)
df_train_cat = df_train_cat[['Et-Balik', 'Kahvaltilik', 'Gida-Sekerleme','Sebze-Meyve',  'Deterjan']]

df_train_cat_pct = df_train_cat.div(df_train_cat.sum(axis=1), axis=0)


from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(n_components=5,random_state=0)
gmm.fit(df_train_cat_pct)
means=gmm.means_

print means
#Train  amixture model with 5 components and display component means

plt.figure(figsize=(10,10))
plt.plot(means.transpose())

plt.xticks(np.arange(5), df_train_cat_pct.columns)
plt.show()
#The components look consistent with cluster centers from KMeans. However, in mixture models each observation consists of a mixture of the components.
#Now let us examine what we did in more detail in order to understand the idea of mixture. We take the customer with index 83 and plot the most likely component.
print(gmm.predict(df_train_cat_pct.iloc[83,:]))
print(gmm.predict_proba(df_train_cat_pct.iloc[83,:]))

component = gmm.predict(df_train_cat_pct.iloc[83,:])[0]

plt.figure(figsize=(10,10))           
plt.plot(means[component,:].transpose(), linewidth=5, label = 'Most likely component')
plt.plot(df_train_cat_pct.iloc[83,:].values, linewidth = 10, label='Customer')
plt.plot(means.transpose())
plt.xticks(np.arange(5), df_train_cat_pct.columns)
plt.legend()
plt.show()

#Unlike KMeans, mixture models allows us to use predict_proba function. We can think of the probabilities that we obtain as the weight of that component in that specific customer profile. We can analyse customer with index 83 to see an example of this. In the example components 1 and 2 are more likely than others (0.66 and 0.34). Let us see whether we can visualize this.
print(gmm.predict_proba(df_train_cat_pct.iloc[83,:]))
# We use probabilities as weights in order to determine the representation of the customer with respect to components.

#Find the representative component by multiplying weights with component centers, use np.dot()
representative = np.dot(gmm.predict_proba(df_train_cat_pct.iloc[83,:]),means)

plt.figure(figsize=(10,10))
plt.plot(means.transpose())
plt.plot(df_train_cat_pct.iloc[83,:].values, linewidth = 10, c='purple', label = 'Customer')
plt.plot(representative.transpose(), linewidth=5, c='k', label = 'Representative')

plt.plot(means[1,:], linewidth = 8, label = ' Component 0, weight = 0.66')
plt.plot(means[2,:], linewidth = 8, label = ' Component 4, weight = 0.34')

plt.xticks(np.arange(5), df_train_cat_pct.columns)
plt.legend()
plt.show()