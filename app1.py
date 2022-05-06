# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 07:50:57 2021

@author: dell
"""

#Importing the Libraries
import numpy as np
from flask import Flask, request,render_template
from flask_cors import CORS
import os
'''from sklearn.externals import joblib'''
import pickle
import flask
import os
import newspaper
from newspaper import Article
import urllib
import requests

#Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')

with open('model.pickle', 'rb') as handle:
	model = pickle.load(handle)

@app.route('/')
def main():
    return render_template('index.html')

#Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict',methods=['GET','POST'])
def predict():
    '''urls=request.form.get("news")
    response=requests.get(urls)
    if response.status_code==200:'''
    url =request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
        #Passing the news article to the model and returing whether it is Fake or Real
    pred = model.predict([news])
    return render_template('index.html', prediction_text='The news is "{}"'.format(pred[0]))
    
    '''else:
        return render_template('index.html', prediction_text='Please enter existing URL')'''
    
if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)