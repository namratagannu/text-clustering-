#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:13:08 2017

@author: namratagannu
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#We read the data , import it because we have the dataset 

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

trainingData = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = 42)

#print ("\n".join(trainingData.data[1].split("\n")[:10]))

#print ("Target is:", trainingData.target_names[trainingData.target[1]])

#converting text to numerical data

countVectorizer = CountVectorizer()
xTrainCounts = countVectorizer.fit_transform(trainingData.data)
print (countVectorizer.vocabulary_.get('software'))


tfidTransformer = TfidfTransformer()
xTrainTfidf = tfidTransformer.fit_transform(xTrainCounts)

#build and fit model 

model = MultinomialNB().fit(xTrainTfidf,trainingData.target)

#use the model to make predictions on new data

new = ['This has nothing to do with church or religion', 'Software engineering is getting hotter']

xNewCounts = countVectorizer.transform(new)

xNewTfidf = tfidTransformer.transform(xNewCounts)

prediction = model.predict(xNewTfidf)

for doc, category in zip(new, prediction):
    print ('%r ---------> %s' %(doc,trainingData.target_names[category]))
    
    


