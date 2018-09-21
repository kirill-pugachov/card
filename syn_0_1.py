# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 09:52:16 2018

@author: tom
"""

import pandas as pd
import itertools
import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVR, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


FILE_NAME = 'train_ds.csv'
#FILE_PATH = 'C:\\Users\\tom\\card_garbage\\Data\\'
FILE_PATH = 'Data/'
TEST_FILE_NAME = 'test_ds.csv'
CITIES_FILE_NAME = 'cities_list.csv'


def get_train_data(FILE_NAME, FILE_PATH):
    temp = pd.read_csv(FILE_PATH + FILE_NAME)
    return temp


def get_categorical_data():
    dtrain = get_train_data(FILE_NAME, FILE_PATH)
    dtest = get_train_data(TEST_FILE_NAME, FILE_PATH)
    return dtrain, dtest


def process_data(df):
    res_list = list()
    id_list = dict(df.clientid.value_counts())
    for client_id in id_list:
        temp = df.loc[df['clientid'] == client_id]
        res_list.append(temp)
    return res_list


def feature_creator(users_list, ccy_tranccy_pairs):
    result_dict = dict([(key,{}) for key in ccy_tranccy_pairs])
    for user in users_list:
        for pair in ccy_tranccy_pairs:
            user_selected = user.query('ccy ==' + str(pair[0]) + ' & ' + 'tranccy == ' + str(pair[1]))
            for unique_mcc in list(user_selected.mcc.unique()):
                sum_by_mcc_user = user_selected[user_selected['mcc'] == unique_mcc]['amount'].sum()
                
                if sum_by_mcc_user > 0:
                
                    if unique_mcc in result_dict[pair].keys():
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair][unique_mcc][1] += sum_by_mcc_user
                        else:
                            result_dict[pair][unique_mcc][0] += sum_by_mcc_user
                    else:
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair].update({unique_mcc:{1: sum_by_mcc_user, 0:0}})
                        else:
                            result_dict[pair].update({unique_mcc:{1:0, 0: sum_by_mcc_user}})
    res = dict((key,value) for key, value in result_dict.items() if len(result_dict[key]) != 0)
    return res


def classifiers_evaluation(df_res, y):

    classifiers = [
    LinearSVC(max_iter=2000),
    LinearSVR(C=0.01, max_iter=2000, dual=False, loss='squared_epsilon_insensitive'),
    KNeighborsClassifier(3),
    SVC(probability=True),
    NuSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=680),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    BernoulliNB(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    MLPClassifier(hidden_layer_sizes=(110, ), max_iter=800),
    SGDClassifier(loss='log', max_iter=800),
    LogisticRegressionCV(max_iter=800)]

    log_cols = ["Classifier", "ROC_AUC score"]
    log = pd.DataFrame(columns=log_cols)

#    quantile = preprocessing.QuantileTransformer(n_quantiles=2500)
#    X = quantile.fit_transform(df_res)
    
#    minmax = preprocessing.MinMaxScaler()
#    X = minmax.fit_transform(df_res)
    
#    norm = preprocessing.Normalizer()
#    X = norm.fit_transform(df_res) 

#    standart = preprocessing.StandardScaler()
#    X = standart.fit_transform(df_res)
    
#    robust = preprocessing.RobustScaler()
#    X = robust.fit_transform(df_res)  
    
    maxabs = preprocessing.MaxAbsScaler()
    X = maxabs.fit_transform(df_res)    

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

    acc_dict = {}

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            train_predictions = clf.predict(X_test)
            acc = roc_auc_score(y_test, train_predictions)

            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc

    for clf in acc_dict:
        acc_dict[clf] = acc_dict[clf] / 10.0
        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
        log = log.append(log_entry)

#    print(acc_dict)
    print(log)
    

if __name__ == '__main__':
    
    df_train, df_test = get_categorical_data()
    
    users_list = process_data(df_train)
    
    ccy_tranccy_pairs = list(itertools.product(list(df_train.ccy.unique()), list(df_train.tranccy.unique())))
    
    result_dict = feature_creator(users_list, ccy_tranccy_pairs)
    
    sexid_1_list = [(key[0], key[1], key_0, value_0[1]) for key, value in result_dict.items() for key_0, value_0 in value.items() if value_0[0] == 0 and value_0[1] != 0]
    
    sexid_0_list = [(key[0], key[1], key_0, value_0[0]) for key, value in result_dict.items() for key_0, value_0 in value.items() if value_0[0] != 0 and value_0[1] == 0]
    
    total_columns_list = copy.deepcopy(sexid_1_list)
    total_columns_list.extend(sexid_0_list)
    
    columns_list = [str(I[0]) + '_' + str(I[1])+ '_' + str(I[2]) for I in total_columns_list]
    
    train_data_df = pd.DataFrame(columns=columns_list, index=range(len(users_list)))
    
    train_data_df['sexid'] = 0
    
    for I in range(0,len(users_list)):
        for U in total_columns_list:
            t = users_list[I].query('ccy ==' + str(U[0]) + ' & ' + 'tranccy ==' + str(U[1]))
            if t[t['mcc'] == U[2]]['amount'].sum():
                print(U[0], U[1], U[2], t[t['mcc'] == U[2]]['amount'].sum())
                train_data_df[str(U[0]) + '_' + str(U[1])+ '_' + str(U[2])][I] = t[t['mcc'] == U[2]]['amount'].sum()
        train_data_df['sexid'][I] = users_list[I].sexid.unique()[0]

    train_data_df.fillna(0, inplace=True)
    
    y_train_full = train_data_df['sexid']
    X_train_full = train_data_df.drop('sexid', axis=1)
    
    classifiers_evaluation(X_train_full, y_train_full)

#    X_train, X_test, y_train, y_test  =  train_test_split(
#                                            X_train_full,
#                                            y_train_full,
#                                            test_size = 0.25,
#                                            random_state=42
#                                            )
#        
#    model_0 = RandomForestClassifier(n_estimators=700, n_jobs=-1)
#    model_0.fit(X_train, y_train)
#    y_pred_forest_short = model_0.predict(X_test)
#    print(classification_report(y_test, y_pred_forest_short))
#    print(roc_auc_score(y_test, y_pred_forest_short))
#
#    param_grid = [
#        {'n_estimators':[640, 660, 680, 700, 1000, 2000, 5000],
#        'max_depth':[1, 2, 3, 4, 5, 10, 20],
#        'bootstrap':[True, False],
#        'max_features':['auto', 'sqrt'],
#        'min_samples_split': [2, 5, 10, 20, 50],
#        'min_samples_leaf': [1, 2, 3, 4, 5]
#        }
#    ]
#    
#    forest_class = RandomForestClassifier()
#    
#    grid_search = GridSearchCV(forest_class, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
#    
#    grid_search.fit(X_train_full, y_train_full)
#    
#    forest_cls = grid_search.best_estimator_
#    
#    print('RandomForest best params', '\n', grid_search.best_params_)
#    
#    forest_cv_res = grid_search.cv_results_
#    
#    for mean_score, params in zip(forest_cv_res['mean_test_score'], forest_cv_res['params']):
#        print(mean_score, params)        
##            train_data_df[str(U[0]) + '_' + str(U[1])+ '_' + str(U[2])][I] = users_list[I][users_list[I]['ccy'] == U[0] and users_list[I]['tranccy'] == U[1] and users_list[I]['mcc'] == U[2]]['amount'].sum()
#        
#        
#    
    
#    result_dict = dict([(key,{}) for key in ccy_tranccy_pairs])
#    
#    for user in users_list:
#        for pair in ccy_tranccy_pairs:
#            user_selected = user.query('ccy ==' + str(pair[0]) + ' & ' + 'tranccy == ' + str(pair[1]))
#            for unique_mcc in list(user_selected.mcc.unique()):
#                sum_by_mcc_user = user_selected[user_selected['mcc'] == unique_mcc]['amount'].sum()
#                
#                if sum_by_mcc_user > 0:
#                
#                    if unique_mcc in result_dict[pair].keys():
#                        if user_selected.sexid.unique()[0] == 1:
#                            result_dict[pair][unique_mcc][1] += sum_by_mcc_user
#                        else:
#                            result_dict[pair][unique_mcc][0] += sum_by_mcc_user
#                    else:
#                        if user_selected.sexid.unique()[0] == 1:
#                            result_dict[pair].update({unique_mcc:{1: sum_by_mcc_user, 0:0}})
#                        else:
#                            result_dict[pair].update({unique_mcc:{1:0, 0: sum_by_mcc_user}})
                        
                    
                
#                dict((user_selected.sexid.unique()[0], user_selected[user_selected['mcc'] == unique_mcc]['amount'].sum()))