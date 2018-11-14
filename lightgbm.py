#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 09:04:02 2018

@author: trgungoe
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import warnings
from IPython.display import display, HTML

df = pd.read_csv('bank-additional-full.csv', delimiter=';')
y= (df['y']=='yes')*1
df.drop('y',axis=1,inplace=True)

df.columns = ['yaş', 'is', 'medeni_durum', 'eğitim', 'gecikme', 'ev', 'borç', 'iletişim', 'ay', 'haftanın_günü',
              'süre', 'kampanya', 'önceki_iletişimden_sonra_geçen_gün', 'iletişim_sayısı', 'iletişim_sonucu', 
              'işsizlik', 'tüketici_fiyat_endeksi', 'tüketici_güven_endeksi', 'euribor_faizi', 'çalışan_sayısı'] 

print (df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
kategorik_sutunlar=['is','medeni_durum','eğitim','gecikme','ev', 'borç', 'iletişim', 'ay',
                      'haftanın_günü', 'iletişim_sonucu']

for i in kategorik_sutunlar:
    le=LabelEncoder()
    df[i]=le.fit_transform(df[i])
print (df.head())

df.drop('süre', inplace = True, axis=1)
df_train, df_test, y_train, y_test = train_test_split(df, y, train_size = 0.7, test_size = 0.3) 

lgb_train = lgb.Dataset(data=df_train, label=y_train,  free_raw_data=False)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

# Kategorik değişkenlerin indeksleri
kategorik_indeks = [1,2,3,4,5,6,7,8,9,13]
print('Kategorik değişkenler: ' + str(df_train.columns[kategorik_indeks].values))

print('Eğitim...')
# Modeli eğitelim
gbm = lgb.train(params,
                lgb_train,
                categorical_feature = kategorik_indeks)
print('Eğitim bitti...')

y_pred = gbm.predict(df_test)

print('Eğri altı alan değeri:', roc_auc_score(y_test, y_pred))
print('İsabetlilik değeri:', accuracy_score(y_test, ( y_pred>= 0.5)*1))



# Değerlendirme veri kümesini oluşturuyoruz.
lgb_eval = lgb.Dataset(data=df_test, label=y_test, reference=lgb_train,  free_raw_data=False)

# Eğitim parametrelerini belirleyelim
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

print('Eğitim...')

# Modeli eğitelim
# Bu sefer değerlendirme veri kümesini de tanıtıyoruz. iteration sayısını 150ye çıkarıyoruz
evals_result={}
gbm = lgb.train(params,
                lgb_train,
                valid_sets = lgb_eval,
                categorical_feature = kategorik_indeks,
                num_boost_round= 150,
                early_stopping_rounds= 25,
                evals_result=evals_result)
print('Eğitim bitti...')

# Tahmin ve değerlendirme
y_pred = gbm.predict(df_test, num_iteration=gbm.best_iteration)

print('En iyi sonucu veren iterasyon: ', gbm.best_iteration)
print('Eğri altı alan değeri:', roc_auc_score(y_test, y_pred))
print('İsabetlilik değeri:', accuracy_score(y_test, ( y_pred>= 0.5)*1))

##AUC
print('Eğri altı alan...')
ax = lgb.plot_metric(evals_result, metric='auc')
ax.set_title('Eğri Altı Alanın İterasyona Göre Değişimi')
ax.set_xlabel('İterasyon')
ax.set_ylabel('Eğri Altı Alan Değeri')
ax.legend_.remove()
plt.show()

##feature importance
ax = lgb.plot_importance(gbm, max_num_features=10)
ax.set_title('')
ax.set_xlabel('Özniteliklerin Önemi')
ax.set_ylabel('Öznitelikler')
plt.show()




###eksik veri ile çalışma
# Veri kümelerinde eksik değerler oluşturuyoruz.
df_train['önceki_iletişimden_sonra_geçen_gün'].replace(999, np.nan, inplace = True)
df_test['önceki_iletişimden_sonra_geçen_gün'].replace(999, np.nan, inplace = True)

print(df_train.isnull().sum())

# Veri kümesi oluşturalım.
lgb_train = lgb.Dataset(data=df_train, label=y_train,  free_raw_data=False)
# Değerlendirme veri kümesini oluşturuyoruz.
lgb_eval = lgb.Dataset(data=df_test, label=y_test, reference=lgb_train,  free_raw_data=False)

# Eğitim parametrelerini belirleyelim
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

# Modeli eğitelim
evals_result={}
gbm = lgb.train(params,
                lgb_train,
                valid_sets = lgb_eval,
                categorical_feature = kategorik_indeks,
                num_boost_round= 150,
                early_stopping_rounds= 25,
                evals_result=evals_result)
print('Eğitim bitti...')

# Tahmin ve değerlendirme
y_pred = gbm.predict(df_test, num_iteration=gbm.best_iteration)

print('En iyi sonucu veren iterasyon: ', gbm.best_iteration)
print('Eğri altı alan değeri:', roc_auc_score(y_test, y_pred))
print('İsabetlilik değeri:', accuracy_score(y_test, ( y_pred>= 0.5)*1))

  
