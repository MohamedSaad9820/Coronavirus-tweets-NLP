#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as nb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("D:\Projects\Crona Classification\Corona_NLP_train.csv",sep=",",encoding="utf-8")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


data=df[["OriginalTweet","Sentiment"]]


# In[7]:


data.head()


# In[8]:


data.info()


# In[9]:


data["Sentiment"].value_counts()


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[11]:


import nltk


# In[12]:


nltk.download("stopwords")


# In[13]:


from nltk.corpus import stopwords


# In[14]:


sw=stopwords.words("english")
sw


# In[15]:


def sw_rm(text):
    filtered_words = []
    for word in text.lower().split():  
        if word not in sw:
            filtered_words.append(word)
    return ' '.join(filtered_words)


# In[16]:


from nltk.stem import PorterStemmer
ps = PorterStemmer()


# In[17]:


def stemming(text):
    stemmed_words = []
    for word in text.lower().split():
        stemmed_words.append(ps.stem(word))
    return ' '.join(stemmed_words)


# In[18]:


import re
def denoise(text):
    return re.sub('[^a-zA-Z0-9 ]+','',text).strip()


# In[19]:


data['clean_tweet'] = data["OriginalTweet"].apply(sw_rm)
#data['clean_tweet'] = data["clean_tweet"].apply(stemming)
#data['clean_text'] = data["clean_tweet"].apply(denoise)
data.head()


# In[20]:


x  = cv.fit_transform(data["clean_tweet"])


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression

x_train,x_test,y_train,y_test = train_test_split(x,data["Sentiment"],test_size=0.20,random_state=110)


# In[22]:


lg = LogisticRegression()


# In[23]:


lg.fit(x_train,y_train)


# In[24]:


round(lg.score(x_train,y_train)*100,2)


# In[25]:


round(lg.score(x_test,y_test)*100,2)


# In[26]:


y_pred=lg.predict(x_test)


# In[27]:


from sklearn.metrics import confusion_matrix , classification_report


# In[28]:


from mlxtend.plotting import plot_confusion_matrix


# In[29]:


conf = confusion_matrix(y_test , y_pred)
conf


# In[30]:


print(classification_report(y_test , y_pred))


# In[ ]:




