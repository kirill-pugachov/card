# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:50:34 2018

@author: tom
"""


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV


FILE_PATH = 'C:\\Users\\tom\\card_garbage\\Data\\'
X_FILE_NAME = 'X_train_full_2809.csv'
Y_FILE_NAME = 'y_train_2809.csv'




if __name__ == '__main__':
    
    
    X_train_full = pd.read_csv(FILE_PATH + X_FILE_NAME, index_col=0)
    y_train = pd.read_csv(FILE_PATH + Y_FILE_NAME, index_col=0, header=None)
    
#    dataset = X_train_full.values
#    
##    X_prom = dataset[:,0:1492].astype(float)
#    X = X_train_full / (X_train_full.max() + 1)
    y = y_train.values

    param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
                  "base_estimator__splitter" :   ["best", "random"],
                  "n_estimators": [10, 20, 30, 40, 50, 75, 100, 150],
                  "base_estimator__max_depth" : [1, 2, 3, 4, 5],
                  "base_estimator__min_samples_split": [5, 6, 7, 8, 9, 10],
                  "base_estimator__min_samples_leaf": [1, 2, 3, 4, 5, 6]         
                 }
    
    
    DTC = DecisionTreeClassifier(random_state = 42, max_features = "auto", class_weight = "balanced", max_depth=2)
    
    ABC = AdaBoostClassifier(base_estimator = DTC)
    
    # run grid search
    grid = GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc')
    
    grid_result = grid.fit(X_train_full, y_train[1])
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
