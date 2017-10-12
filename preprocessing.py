# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:31:38 2017

@author: AASHIMA SINGH
"""
import string
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
train_x = pd.read_csv('train_set_x.csv',encoding="utf-8")
train_y = pd.read_csv('train_set_y.csv')
print(train_x.size)
#remove empty cells
train_x['Text'].replace('', np.nan, inplace=True)
train_x.dropna(subset=['Text'], inplace=True)
print(train_x.size)
#remove numbers
train_x['Text'] = train_x['Text'].str.replace('\d+', '')
print(train_x.size)
#remove words starting with https
train_x['Text'] = train_x['Text'].str.replace('https\w+', '')

train_x['Text'] = train_x['Text'].str.replace('½','')
train_x['Text'] = train_x['Text'].str.replace('¾','')
train_x['Text'] = train_x['Text'].str.replace('¼','')
train_x['Text'] = train_x['Text'].str.replace(r'([^\s\w]|_)+','')
#removing non-latin characters
train_x['Text'] = train_x['Text'].str.replace(u'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]','')


#remove extra white space
train_x['Text'] = train_x['Text'].apply(lambda col: col.strip())

merged_df = pd.merge(train_y,train_x,on='Id')

vectorizer = TfidfVectorizer(analyzer='char')
X = vectorizer.fit_transform(train_x['Text'].values).toarray()
print(X.shape)
print(vectorizer.get_feature_names());
np.savetxt("id_vector.csv",X,delimiter=",")
