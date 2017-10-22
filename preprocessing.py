# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:31:38 2017

@author: AASHIMA SINGH
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

train_x = pd.read_csv('train_set_x.csv',encoding="utf-8")
test_x = pd.read_csv('test_set_x.csv', encoding ="utf-8")
train_y = pd.read_csv('train_set_y.csv')
print(train_x.size)

train_x['Text'].replace('', np.nan, inplace=True)
train_x['Text'].replace(np.nan, '0' , inplace=True)
test_x['Text'].replace(np.nan, '0' , inplace=True)
#print(test_x.size)

#remove numbers
train_x['Text'] = train_x['Text'].str.replace('\d+', '')
test_x['Text'] = test_x['Text'].str.replace('\d+', '')

#remove words starting with https
train_x['Text'] = train_x['Text'].str.replace('https\w+', '')
test_x['Text'] = test_x['Text'].str.replace('https\w+', '')

#remove fractions
train_x['Text'] = train_x['Text'].str.replace(u'[\u00BC\u00BE\u00BD]','')
test_x['Text'] = test_x['Text'].str.replace(u'[\u00BC\u00BE\u00BD]','')

#remove superscripts
train_x['Text'] = train_x['Text'].str.replace(u'[\xb9\xb2\xb3\u2070]','')
test_x['Text'] = test_x['Text'].str.replace(u'[\xb9\xb2\xb3\u2070]','')

train_x['Text'] = train_x['Text'].str.replace(r'([^\s\w]|_)+','')
test_x['Text'] = test_x['Text'].str.replace(r'([^\s\w]|_)+','')

#removing non-latin characters
train_x['Text'] = train_x['Text'].str.replace(u'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]','')
test_x['Text'] = test_x['Text'].str.replace(u'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]','')

#remove extra white space
train_x['Text'] = train_x['Text'].apply(lambda col: col.strip())
test_x['Text'] = test_x['Text'].apply(lambda col: col.strip())

merged_df = pd.merge(train_y,train_x,on='Id')
#new train y with fewer rows
new_train_y = merged_df.iloc[:,:2]

vectorizer1 = CountVectorizer(analyzer='char')
vectorizer2 = CountVectorizer(analyzer='char')
X = vectorizer1.fit_transform(train_x['Text'].values).toarray()
Xtest = vectorizer2.fit_transform(test_x['Text'].values).toarray()
print(X.shape)
print(Xtest.shape)
print(vectorizer1.get_feature_names())
print(vectorizer2.get_feature_names())

feature_train = vectorizer1.get_feature_names()
feature_test = vectorizer2.get_feature_names()

posd = []
difference = list(set(feature_test) - set(feature_train))
for d in difference:
    posd.append(feature_test.index(d))
Xtest = np.delete(Xtest,posd,1)

for f in feature_train:
    if f not in feature_test:
        pos = feature_train.index(f)
        #print(pos)
        if pos<(len(Xtest[0])):
            Xtest = np.insert(Xtest,pos,0,axis=1)
xsize = len(Xtest)
zerocols = np.zeros((xsize,1),dtype=np.int);
Xtest = np.concatenate((Xtest,zerocols),axis=1)

#forming vectors of 0s and 1s
#X[X>0] = 1;
#Xtest[Xtest>0] = 1;

#add column of ID to 2D array
col = np.array(train_x['Id'])
col2 = np.array(test_x['Id'])
Y = np.column_stack((col,X[:,1:-1]))
Ytest = np.column_stack((col2,Xtest[:,1:]))
print(Y.shape)
np.savetxt("id_vector_train.csv",Y,delimiter=",")
np.savetxt("id_vector_test.csv",Ytest,delimiter=",")
