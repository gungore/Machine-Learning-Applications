# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 13:29:09 2017

@author: Erdig
"""

import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
print(string.punctuation)
stopwords_en = pd.read_csv('C:/Users/Erdig/Desktop/sabanci/Practical Case Studies/1.ders/english.txt')
print(stopwords_en.head())

title = 'Studying customer loyalty: An analysis for an online retailer in Turkey'
print(title)

title = title.lower()
print(title)

title = title.translate(None,string.punctuation)
print(title)


title = [word for word in title.split() if word not in stopwords_en.values ]

print(title)
#We convert the text to lower case (lower), remove punctuation (translate, string.punctuation), and remove stop words


%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#We load the data on publishing of Turkish academics. 
df_paper = pd.read_csv('C:/Users/Erdig/Desktop/sabanci/Practical Case Studies/1.ders/papers.csv')
df_affiliation = pd.read_csv('C:/Users/Erdig/Desktop/sabanci/Practical Case Studies/1.ders/Affiliations.csv')

print(df_paper.head())
print(df_affiliation.head())

print df_paper['paperID']
print df_paper.iloc[0:3,0:2]  #0,1,2 satır ve 0,1 kolon
print df_paper[df_paper['year']>=2005]['paperID']

#Let us see and how many papers use the word data on their titles since 1997 (-20 years) and do the same for neural network.
#str.contains() can be used to check the existence of a string in another string
print(df_paper['titleClean'].str.contains('data').sum())
df_data = df_paper[df_paper['titleClean'].str.contains('data')]
df_data = df_data[df_data['year'] >= 1997]
print(df_data)

a,b = np.unique(df_data['year'], return_counts=True)

print(a)
print(b)

plt.plot(a,b)
plt.show()

print(df_paper['titleClean'].str.contains('neural network').sum())
df_data = df_paper[df_paper['titleClean'].str.contains('neural network')]
df_data = df_data[df_data['year'] >= 1997]
print(df_data)

a_nn,b_nn = np.unique(df_data['year'], return_counts=True)
print(a)
print(b)

plt.plot(a_nn,b_nn)
plt.show()

#iki grafik ortak y değeri

f,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5),sharey=True)
ax1.plot(a,b)
plt.sca(ax1)

ax1.set_title('Number of papers with the word \'data\' on the title')
plt.sca(ax2)


ax2.plot(a_nn,b_nn)

ax2.set_title('Number of papers with the word \'neural network\' on the title')
plt.show()


#TERM DOCUMENT FREQUENCY
#We create and examine tf-idf matrix and count matrix using the modules below.
#There is a huge number of documents, so we can use the first 1000.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

df_paper1 = df_paper.iloc[0:1000,:]
tdm = TfidfVectorizer(min_df=20, stop_words='english')
tdm.fit(df_paper1['titleClean'])
tfidf=tdm.transform(df_paper1['titleClean'])
print tfidf

cv = CountVectorizer(min_df=20,stop_words='english')
count=cv.fit_transform(df_paper['titleClean'])
print count

#We apply KMeans algorithm on tf-idf matrix in order to find similar documents. The number of clusters are set arbitrarily, you can experiment with the number of clusters to see if the results are robus#
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 60, n_init = 10, random_state=1)

kmeans.fit(tfidf)

labels = kmeans.predict(tfidf)

print labels==30
print df_paper1[labels==30]['titleClean']

##Correlation heat-map
df_ptd = pd.DataFrame(count.todense())

vocabulary = dict((v, k) for k, v in cv.vocabulary_.iteritems())
df_ptd.columns=vocabulary.items()
corr_mat = df_ptd.corr()
print corr_mat
# We draw correlation heat map
import seaborn as sns
sns.set(context="paper", font="arial", style = "whitegrid", font_scale=2.0)
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

# Draw the heatmap using seaborn and heatmap function
f.tight_layout()
plt.title('Correlation Map')



#FURTHER APPLICATION
print(df_affiliation[df_affiliation.PaperID == '0000197A'])
print(df_paper[df_paper.paperID == '0000197A'])

print(df_affiliation.columns)
print(df_paper.columns)

df_affiliation.columns=['paperID', 'authorID', 'affiliationID']

df_merge = df_affiliation.merge(df_paper, how='inner', on='paperID')

print df_merge[df_merge['paperID']=='0000197A']


