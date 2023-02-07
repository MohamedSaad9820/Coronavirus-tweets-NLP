#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask,jsonify,request
from cronaclassifier import *
app      = Flask(__name__)
df       = get_data()
x        = preprocess(df)
accuracy = train(x,df) # train and evaluation when app started

@app.route('/')        ## Homepage
def home():
    data = {
        'name':'Crona_Classification',
        'page_name':'الصفحة الرئيسيه'
    }
    return jsonify(data)

@app.route('/train') ## train
def fl_train():
    accuracy = train(x,df)
    data = {
        'name':'Crona_Classification',
        'page_name':'Train',
        'accuracy':accuracy
    }
    return jsonify(data)

@app.route('/predict') ## predict
def fl_predict():
    text  = request.args['text']
    print(text)
    label = predict(text)
    data = {
        'name':'Crona_Classification',
        'page_name':'Prediction',
        'crona result':label
    }
    return jsonify(data)
app.run()


# In[ ]:




