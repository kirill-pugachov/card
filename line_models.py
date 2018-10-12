# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:56:03 2018

@author: tom
"""


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pandas as pd
import numpy as np


#FILE_PATH = 'Data/'
FILE_PATH = 'C:\\Users\\tom\\card_garbage\\Data\\'
X_FILE_NAME = 'new_dataframe_1.csv'


def LogisticRegression_tuning(X_train, y_train):
    '''
    Проводит первичный тюнинг LogisticRegression
    '''

    param_grid = [
        {'penalty':['l2'],
         'class_weight':[None, 'balanced'],
         'fit_intercept':[False, True],
         'dual':[False],
         'max_iter': [50, 75, 100, 200, 400, 800],
         'tol':[0.00001, 0.0001, 0.001],
         'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 200, 500, 1000],
         'solver':['newton-cg', 'sag', 'lbfgs']
        }
    ]

    lin_class = LogisticRegression()

    grid_search = GridSearchCV(lin_class, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)

    lin_cls = grid_search.best_estimator_

    print('LogisticRegression best params', '\n', grid_search.best_params_)
    print('LogisticRegression best score', '\n', grid_search.best_score_)

    return lin_cls


def linear_SVC_tuning(X_train, y_train):
    '''
    Проводит первичный тюнинг linear_SVC
    '''

    param_grid = [
        {'penalty':['l1'],
         'class_weight':[None, 'balanced'],
         'fit_intercept':[False, True],
         'dual':[False],
         'max_iter': [1000, 2000, 5000, 10000],
         'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
         'loss':['squared_hinge']
        },
        {'penalty':['l2'],
         'class_weight':[None, 'balanced'],
         'fit_intercept':[False, True],
         'dual':[True, False],
         'max_iter': [1000, 2000, 5000, 10000],
         'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
         'loss':['squared_hinge']
        },
    ]

    lin_class = LinearSVC()

    grid_search = GridSearchCV(lin_class, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)

    lin_cls = grid_search.best_estimator_

    print('linear_SVC best params', '\n', grid_search.best_params_)
    print('linear_SVC best score', '\n', grid_search.best_score_)

    return lin_cls


def random_forest_tuninig(X_train_full, y_train_full):
    param_grid = [
        {'n_estimators':[30, 50, 75, 100],
         'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
         'bootstrap':[True, False],
         'max_features':['auto', 'sqrt'],
         'min_samples_split': [5, 6, 7, 8],
         'min_samples_leaf': [1, 2, 3, 4]
        }
    ]

    forest_class = RandomForestClassifier()

    grid_search = GridSearchCV(
        forest_class,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2)

    grid_search.fit(X_train_full, y_train_full)

    forest_cls = grid_search.best_estimator_

    print('RandomForest best params', '\n', grid_search.best_params_)
    print('RandomForest best score', '\n', grid_search.best_score_)

#    forest_cv_res = grid_search.cv_results_
#
#    for mean_score, params in zip(forest_cv_res['mean_test_score'], forest_cv_res['params']):
#        print(mean_score, params)

    return forest_cls



if __name__ == '__main__':
    
    
    X_train_full = pd.read_csv(FILE_PATH + X_FILE_NAME, index_col=0)
    y_train = X_train_full['sexid']
    
    X_train_full.drop(['sexid'], axis=1, inplace=True)
    X_train_full = X_train_full / (X_train_full.values.max() + 1)
#    X_train_full = X_train_full + 0.000001
    
    random_forest_tuninig(X_train_full, y_train)
    
    linear_SVC_tuning(X_train_full, y_train)
    
    LogisticRegression_tuning(X_train_full, y_train)