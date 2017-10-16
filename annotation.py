# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

train_x = pd.read_csv('train_set_x.csv',encoding="utf-8")
test_x = pd.read_csv('test_set_x.csv', encoding ="utf-8")
train_y = pd.read_csv('train_set_y.csv')
merged_df = pd.merge(train_y,train_x,on='Id')
merged_df = merged_df.sort_values(['Category'],ascending=1)
merged_df.reset_index(inplace=True,drop=True)
print(train_x.size)

merged_df['Text'].replace(np.nan, '',inplace=True)
#splitting data according to languages
df_slovak = merged_df.loc[merged_df['Category']==0]
df_slovak['Id'] = df_slovak.index
df_french = merged_df.loc[merged_df['Category']==1].reset_index(drop=True)
df_french['Id'] = df_french.index + 14167
df_spanish = merged_df.loc[merged_df['Category']==2].reset_index(drop=True)
df_spanish['Id'] = df_spanish.index + df_french['Id'].iloc[-1] +1
df_german = merged_df.loc[merged_df['Category']==3].reset_index(drop=True)
df_german['Id'] = df_german.index + df_spanish['Id'].iloc[-1] +1
df_polish = merged_df.loc[merged_df['Category']==4].reset_index(drop=True)
df_polish['Id'] = df_polish.index + df_german['Id'].iloc[-1] +1

ndf = pd.DataFrame(0, index=np.arange(len(df_slovak)*2-2), columns=['Id','Category','Text'])
ndf.loc[0:14166,:]= df_slovak;
french_size = len(df_french)-1
slovak_size = len(df_slovak)-1
amplification_ratio = french_size/slovak_size;
idx = len(df_slovak)-1
for i in range(len(df_slovak)-1):
    array = np.random.permutation(slovak_size)
    equal=0;
    if i%1000==0:
        print(i)
    for j in range(len(array)):
        if j-equal>amplification_ratio:
            break
        if j==i:
            equal=1;
            continue
        idx =idx+1
        new_text = df_slovak.loc[i,'Text']+" " + df_slovak.loc[j,'Text']
        ndf.loc[idx,:] = [idx,0,new_text]
  
ndf2 = pd.DataFrame(0, index=np.arange(len(df_spanish)*2-2), columns=['Id','Category','Text'])
ndf2.loc[0:69973,:]= df_spanish;
spanish_size = len(df_spanish)-1
amplification_ratio = french_size/spanish_size;
idx = spanish_size
for i in range(spanish_size):
    array = np.random.permutation(spanish_size)
    equal=0;
    if i%1000==0:
        print(i)
    for j in range(len(array)):
        if j-equal>amplification_ratio:
            break
        if j==i:
            equal=1;
            continue
        idx =idx+1
        new_text = df_spanish.loc[i,'Text']+" " + df_spanish.loc[j,'Text']
        ndf2.loc[idx,:] = [idx,0,new_text]
        
ndf3 = pd.DataFrame(0, index=np.arange(len(df_german)*2-2), columns=['Id','Category','Text'])
ndf3.loc[0:37013,:]= df_german;
german_size = len(df_german)-1
amplification_ratio = french_size/german_size;
idx = len(df_german)-1
for i in range(german_size):
    array = np.random.permutation(german_size)
    equal=0;
    if i%1000==0:
        print(i)
    for j in range(len(array)):
        if j-equal>amplification_ratio:
            break
        if j==i:
            equal=1;
            continue
        idx =idx+1
        new_text = df_german.loc[i,'Text']+" " + df_german.loc[j,'Text']
        ndf3.loc[idx,:] = [idx,0,new_text]
        
ndf4 = pd.DataFrame(0, index=np.arange(len(df_polish)*2-2), columns=['Id','Category','Text'])
ndf4.loc[0:14166,:]= df_polish;
polish_size = len(df_polish)-1
amplification_ratio = french_size/polish_size;
idx = len(df_polish)-1
for i in range(polish_size):
    array = np.random.permutation(polish_size)
    equal=0;
    if i%1000==0:
        print(i)
    for j in range(len(array)):
        if j-equal>amplification_ratio:
            break
        if j==i:
            equal=1;
            continue
        idx =idx+1
        new_text = df_polish.loc[i,'Text']+" " + df_polish.loc[j,'Text']
        ndf4.loc[idx,:] = [idx,0,new_text]
        
finaltrain = pd.concatenate([ndf,df_french,ndf2,ndf3,ndf4])
    
#remove empty cells
finaltrain['Text'].replace('', np.nan, inplace=True)
finaltrain.dropna(subset=['Text'], inplace=True)
test_x.dropna(subset=['Text'], inplace=True)
#print(test_x.size)

#remove numbers
finaltrain['Text'] = finaltrain['Text'].str.replace('\d+', '')
test_x['Text'] = test_x['Text'].str.replace('\d+', '')

#remove words starting with https
finaltrain['Text'] = finaltrain['Text'].str.replace('https\w+', '')
test_x['Text'] = test_x['Text'].str.replace('https\w+', '')

#remove fractions
finaltrain['Text'] = finaltrain['Text'].str.replace(u'[\u00BC\u00BE\u00BD]','')
test_x['Text'] = test_x['Text'].str.replace(u'[\u00BC\u00BE\u00BD]','')

#remove superscripts
finaltrain['Text'] = finaltrain['Text'].str.replace(u'[\xb9\xb2\xb3\u2070]','')
test_x['Text'] = test_x['Text'].str.replace(u'[\xb9\xb2\xb3\u2070]','')

finaltrain['Text'] = finaltrain['Text'].str.replace(r'([^\s\w]|_)+','')
test_x['Text'] = test_x['Text'].str.replace(r'([^\s\w]|_)+','')

#removing non-latin characters
finaltrain['Text'] = finaltrain['Text'].str.replace(u'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]','')
test_x['Text'] = test_x['Text'].str.replace(u'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]','')

#remove extra white space
finaltrain['Text'] = finaltrain['Text'].apply(lambda col: col.strip())
test_x['Text'] = test_x['Text'].apply(lambda col: col.strip())

#merged_df = pd.merge(train_y,train_x,on='Id')
#new train y with fewer rows
#new_train_y = merged_df.iloc[:,:2]

vectorizer1 = CountVectorizer(analyzer='char')
vectorizer2 = CountVectorizer(analyzer='char')
X = vectorizer1.fit_transform(finaltrain['Text'].values).toarray()
Xtest = vectorizer2.fit_transform(test_x['Text'].values).toarray()
#print(X.shape)
#print(Xtest.shape)
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
        print(pos)
        if pos<(len(Xtest[0])):
            Xtest = np.insert(Xtest,pos,0,axis=1)
xsize = len(Xtest)
zerocols = np.zeros((xsize,1),dtype=np.int);
Xtest = np.concatenate((Xtest,zerocols),axis=1)

#forming vectors of 0s and 1s
X[X>0] = 1;
Xtest[Xtest>0] = 1;

#add column of ID to 2D array
col = np.array(finaltrain['Id'])
col2 = np.array(test_x['Id'])
Y = np.column_stack((col,X[:,1:-1]))
Ytest = np.column_stack((col2,Xtest[:,1:]))
print(Y.shape)
np.savetxt("id_vector_train.csv",Y,delimiter=",")
np.savetxt("id_vector_test.csv",Ytest,delimiter=",")