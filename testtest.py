

import argparse
import numpy as np
import os
import pickle

#from sklearn.svm import SVC
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
    parser.add_argument("--output", type=str, default="data/svm_C1_c200.p")

    args = parser.parse_args()
    dataset_bow = args.dataset_bow
   # C = args.C
    output = args.output

    # Load and make dataset.
    dataset = pickle.load(open(dataset_bow, "rb"))
    X, Y = make_dataset(dataset)


    
    
    from sklearn.grid_search import GridSearchCV
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
# Build a classification task using 3 informative features
    X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)


    rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

    param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
    }

    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X, y)
    
pickle.dump(clf, open(output, "wb"))