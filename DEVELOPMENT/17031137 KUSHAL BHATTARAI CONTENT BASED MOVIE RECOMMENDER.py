#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval


# In[3]:


ds = pd.read_csv(os.getcwd()+'\\sample data\\sampledata.csv')


# In[4]:


features = ['keywords','genres']
for feature in features:
    ds[feature] = ds[feature].apply(literal_eval)


# In[5]:


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 1:
            names = names[:50]
        setname=set(names)
        listname=list(setname)
        return listname
    return []


# In[6]:


for feature in features:
    ds[feature] = ds[feature].apply(get_list)


# In[7]:


def create_key(x):
    return ' '.join(x['genres']) +' '+ ' '.join(x['keywords']) 
ds['key'] = ds.apply(create_key, axis=1)


# In[8]:



tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
ds["overview"].fillna(" ", inplace = True) 
ds["title"].fillna(" ", inplace = True) 
ds["genres"].fillna(" ", inplace = True) 
ds["keywords"].fillna(" ", inplace = True) 
ds["tagline"].fillna(" ", inplace = True) 
ds['key']=ds['key']+' '+ds['title']+' '+ds['tagline']+' '+ds['overview']

tfidf_matrix = tf.fit_transform(ds['key'])


# In[9]:


cos_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[10]:


result_dict = {}
for key, row in ds.iterrows():
    similar_ind = cos_similarities[key].argsort()[:-50:-1]
    similar_items = [(cos_similarities[key][i], ds['id'][i]) for i in similar_ind]
    result_dict[row['id']] = similar_items[1:]


# In[11]:


def item(id):
    return ds.loc[ds['id'] == id]['title'].tolist()[0]


# In[12]:


def recommend(id):
    num=5
    print("Movie "+ item(id) )
    print(' ')
    recs = result_dict[id][:num]
    for rec in recs:
        if rec[0] > 0: 
            print(item(rec[1]))
            


# In[13]:


recommend(id= np.random.randint(1,len(cos_similarities),1)[0])


# In[14]:


recommend(id= np.random.randint(1,len(cos_similarities),1)[0])


# In[15]:


recommend(1)


# In[16]:


titles = ds['title']
indices = pd.Series(ds.index, index=ds['title'])
def get_recommendations(title):
    print('Movie ' +title )
    idx = indices[title]
    sim_scores = list(enumerate(cos_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[17]:


get_recommendations('Avatar')



