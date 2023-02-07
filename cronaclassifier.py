#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as nb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
sw=stopwords.words("english")
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import re
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
lg = LogisticRegression()


def get_data():
    df=pd.read_csv("D:\Projects\Crona Classification\Corona_NLP_train.csv",sep=",",encoding="utf-8")
    return df

def sw_rm(text):
    filtered_words = []
    for word in text.lower().split():  
        if word not in sw:
            filtered_words.append(word)
    return ' '.join(filtered_words)

def stemming(text):
    stemmed_words = []
    for word in text.lower().split():
        stemmed_words.append(ps.stem(word))
    return ' '.join(stemmed_words)

def denoise(text):
    return re.sub('[^a-zA-Z0-9 ]+','',text).strip()


def preprocess(df):
    data=df[["OriginalTweet","Sentiment"]]
    data['clean_tweet'] = data["OriginalTweet"].apply(sw_rm)
    data['clean_tweet'] = data["clean_tweet"].apply(stemming)
    data['clean_text'] = data["clean_tweet"].apply(denoise)
    x  = cv.fit_transform(data["clean_tweet"])
    return x


def train(x,data):
    x_train,x_test,y_train,y_test = train_test_split(x,data["Sentiment"],test_size=0.20,random_state=110)
    lg.fit(x_train,y_train)
    return round(lg.score(x_train,y_train)*100,2)

def predict(test):
    test = sw_rm(test)
    test = stemming(test)
    test = denoise(test)
    test = cv.transform([test])
    return lg.predict(test)[0]


# In[ ]:




