# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 04:23:29 2019

@author: CCF
"""

import argparse
import numpy as np
import os
import pickle

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

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
    parser.add_argument("--output", type=str, default="data/svm_C1_c200.p")

    args = parser.parse_args()
    dataset_bow = args.dataset_bow
    C = args.C
    output = args.output

    # Load and make dataset.
    dataset = pickle.load(open(dataset_bow, "rb"))
    X, Y = make_dataset(dataset)

    # Train SVM and save to file.
    """clf = SVC(C=C, kernel="linear", verbose=True)
    clf.fit(X, Y)"""
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
    test_results = []
    for estimator in n_estimators:
        clf= RandomForestClassifier(n_estimators=32)
        clf.fit(X,Y)
        for video in dataset:
            predicted = clf.predict([video["features"]])
            false_positive_rate, true_positive_rate, thresholds = roc_curve(video["category"],predicted)
            
            
            """roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)"""
        
            #pickle.dump(clf, open(output, "wb"))
           # CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", 
       # "walking"]
           