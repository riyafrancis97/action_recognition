# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:29:40 2019

@author: CCF
"""

import argparse
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
def make_dataset(dataset):
    X = []
    Y = []

    for video in dataset:
        X.append(video["features"])
        Y.append(video["category"])

    return X, Y

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", 
    "walking"]
if __name__ == "__main__":
    confusion_matrix = np.zeros((6, 6))
    parser = argparse.ArgumentParser(description="Train SVM on bow vectors")
    parser.add_argument("--dataset_bow", type=str, default="data/train_bow_c200.p",
        help="path to dataset bow file")
    parser.add_argument("--C", type=float, default=1)
    parser.add_argument("--output", type=str, default="data/svm_C1_c500.p")

    args = parser.parse_args()
    dataset_bow = args.dataset_bow
    C = args.C
    output = args.output

    # Load and make dataset.
    #dataset = pickle.load(open(dataset_bow, "rb"))
    #,Y= make_dataset(dataset)
    #kmeans = KMeans(n_clusters=2)
    #kmeans.fit(X)
    #clf = SVC(C=C, kernel="linear", verbose=True)
    #clf.fit(X,kmeans.labels_) 

  #
    
     # Load and make dataset.
    dataset = pickle.load(open(dataset_bow, "rb"))
    X, Y = make_dataset(dataset)

    # Train SVM and save to file.
    #clf = SVC(C=C, kernel="linear", verbose=True)
    #lf.fit(X, Y)
    #clf = MultinomialNB().fit(X,Y)
    
    
    text_clf = Pipeline([('clf', MultinomialNB())])
    
    
    
    parameters = {
              'clf__alpha': (1e-2, 1e-3)}
    
   gs_clf = GridSearchCV(text_clf,parameters,cv=6,iid=False,n_jobs=-1)
   gs_clf = gs_clf.fit(X,Y)
    
    data = pickle.load(open("data/train_bow_c200.p", "rb"))
    correct = 0
    for video in data:
        predicted = gs_clf.predict([video["features"]])


        # Check if majority is correct.
        if predicted == video["category"]:
            correct += 1

        confusion_matrix[CATEGORIES.index(predicted),
            CATEGORIES.index(video["category"])] += 1

    print("%d/%d Correct" % (correct, len(data)))
    print("Accuracy =", correct / len(data))
    
    print("Confusion matrix")
print(confusion_matrix)

  
   
   
    
    
    
    
    
    
    
    
    
    #data = pickle.load(open("data/test_bow_c500.p", "rb"))
   #pickle.dump(clf, open(output, "wb"))
    
    #parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               #'tfidf__use_idf': (True, False),
               #'clf__alpha': (1e-2, 1e-3)}
    
"""kmeans2 = KMeans(n_clusters=2)
    kmeans2.fit(s)
    clf = SVC(C=C, kernel="linear", verbose=True)
    clf.fit(s,kmeans2.labels_) 
   
    kmeans3 = KMeans(n_clusters=2)
    kmeans3.fit(n)
    clf = SVC(C=C, kernel="linear", verbose=True)
    clf.fit(n,kmeans3.labels_) 
    
    
    
    kmeans4 = KMeans(n_clusters=2)
    kmeans4.fit(m)
    clf = SVC(C=C, kernel="linear", verbose=True)
    clf.fit(m,kmeans4.labels_) 
    
    kmeans5 = KMeans(n_clusters=2)
    kmeans5.fit()
    clf = SVC(C=C, kernel="linear", verbose=True)
    clf.fit(s,kmeans.labels_)
    o = []
    p = []
    for i in kmeans.labels_:
        if (i==0):
            o.append(s[i])
        else:
            p.append(s[i])     
            
    kmeans6 = KMeans(n_clusters=2)
    kmeans4.fit(o)
    clf = SVC(C=C, kernel="linear", verbose=True)
    clf.fit(o,kmeans4.labels_)  
    
    kmeans7 = KMeans(n_clusters=2)
    kmeans4.fit(p)
    clf = SVC(C=C, kernel="linear", verbose=True)
    clf.fit(p,kmeans4.labels_) 
    """
    
  