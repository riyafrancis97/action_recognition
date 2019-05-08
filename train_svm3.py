# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:52:08 2019

@author: CCF
"""

import argparse
import numpy as np
import os
import pickle

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier

def make_dataset(dataset):
    X = []
    Y = []

    for video in dataset:
        X.append(video["features"])
        Y.append(video["category"])

    return X, Y

if __name__ == "__main__":
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
    dataset = pickle.load(open(dataset_bow, "rb"))

    X, Y = make_dataset(dataset)
    """svm = SVC(C=C, kernel="linear", verbose=True)
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    model = GridSearchCV(svm, param_grid, iid=False,cv=10)
    model.fit(X, Y)
    print(model.best_params_)"""
        

    # Train SVM and save to file.
#   from sklearn.cluster import AgglomerativeClustering

    """cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(X)  
    print(Y)

    svm = SVC(kernel='rbf',C=7,gamma=0.000095, verbose=True)
    print(cluster.labels_)
    svm.fit(cluster.labels_,Y)"""
    
    clf = RandomForestClassifier(n_estimators=160)
    clf.fit(X,Y)
    
pickle.dump(svm, open(output, "wb"))