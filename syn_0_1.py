# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 09:52:16 2018

@author: tom
"""

import pandas as pd
import numpy as np
import itertools
import copy
import datetime
import csv

#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
#from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

#from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVR, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

FILE_NAME = 'train_ds.csv'
FILE_PATH = 'C:\\Users\\tom\\card_garbage\\Data\\'
#FILE_PATH = 'Data/'
TEST_FILE_NAME = 'test_ds.csv'
CITIES_FILE_NAME = 'cities_list.csv'

def get_train_data(FILE_NAME, FILE_PATH):
    temp = pd.read_csv(FILE_PATH + FILE_NAME)
    return temp

def get_categorical_data():
    dtrain = get_train_data(FILE_NAME, FILE_PATH)
    dtest = get_train_data(TEST_FILE_NAME, FILE_PATH)
    return dtrain, dtest


#def process_data(df):
#    res_list = list()
#    id_list = dict(df.clientid.value_counts())
#    for client_id in id_list:
#        temp = df.loc[df['clientid'] == client_id]
#        res_list.append(temp)
#    return res_list
    

def process_data(df):
    res_list = list()
    id_list = dict(df.clientid.value_counts())
    for client_id in id_list:
        temp = df.loc[df['clientid'] == client_id]
        year_list = dict(df.year.value_counts())
        for year in year_list:
            temp_year = temp.loc[temp['year'] == year]
            if not temp_year.empty:
                month_list = dict(temp_year.month.value_counts())
                for month in month_list:
                    temp_month = temp_year.loc[temp_year['month'] == month]
                    if not temp_month.empty:
                        day_list = dict(temp_month.day.value_counts())
                        for day in day_list:
                            temp_day = temp_month.loc[temp_month['day'] == day]
                            if not temp_day.empty:
                                res_list.append(temp_day)
    return res_list

#
#def process_data(df):
#    res_list = list()
#    id_list = dict(df.clientid.value_counts())
#    for client_id in id_list:
#        temp = df.loc[df['clientid'] == client_id]
#        year_list = dict(df.year.value_counts())
#        for year in year_list:
#            temp_year = temp.loc[temp['year'] == year]
#            if not temp_year.empty:
#                month_list = dict(temp_year.month.value_counts())
#                for month in month_list:
#                    temp_month = temp_year.loc[temp_year['month'] == month]
#                    if not temp_month.empty:
#                        res_list.append(temp_month)
#    return res_list


country_dict = {'GE': ['GE', 'GEGE'],
 'MY': ['MY', 'MYMY'],
 'AU': ['AU', 'AUAU'],
 'CN': ['CN', 'CNCN'],
 'KZ': ['KZ', 'KZKZ'],
 'MK': ['MK', 'MKMK'],
 'CH': ['CH', 'CHCH'],
 'EG': ['EG', 'EGEG'],
 'NL': ['NL', 'NLNL'],
 'DO': ['DO', 'DODO'],
 'IT': ['IT', 'ITIT'],
 'DE': ['DE', 'DEDE'],
 'SE': ['SE', 'SESE'],
 'TR': ['TR', 'TRTR'],
 'GR': ['GR', 'GRGR'],
 'ME': ['ME', 'MEME'],
 'SI': ['SI', 'SISI'],
 'IN': ['IN', 'ININ'],
 'NO': ['NO', 'NONO'],
 'RU': ['RU', 'RURU'],
 'BY': ['BY', 'BYBY'],
 'IE': ['IE', 'IEIE'],
 'RO': ['RO', 'RORO'],
 'CY': ['CY', 'CYCY'],
 'SK': ['SK', 'SKSK'],
 'DK': ['DK', 'DKDK'],
 'HK': ['HK', 'HKHK'],
 'HU': ['HU', 'HUHU'],
 'AE': ['AE', 'AEAE'],
 'RS': ['RS', 'RSRS'],
 'PL': ['PL', 'PLPL'],
 'ES': ['ES', 'ESES'],
 'LV': ['LV', 'LVLV'],
 'MC': ['MC', 'MCMC'],
 'GB': ['GB', 'GBGB'],
 'IL': ['IL', 'ILIL'],
 'CZ': ['CZ', 'CZCZ'],
 'MU': ['MU', 'MUMU'],
 'BE': ['BE', 'BEBE'],
 'AT': ['AT', 'ATAT'],
 'FR': ['FR', 'FRFR'],
 'PT': ['PT', 'PTPT'],
 'HR': ['HR', 'HRHR'],
 'AD': ['AD', 'ADAD'],
 'GI': ['GI', 'GIGI'],
 'BG': ['BG', 'BGBG'],
 'MD': ['MD', 'MDMD'],
 'LU': ['LU', 'LULU'],
 'UA': ['. UA', 'A UA', 'K UA', 'SKUA', 'UAUA', 'UA'],
 'US': ['ARUS',
  'AZUS',
  'CAUS',
  'DCUS',
  'DEUS',
  'FLUS',
  'GAUS',
  'IAUS',
  'IDUS',
  'ILUS',
  'KSUS',
  'MDUS',
  'MNUS',
  'MOUS',
  'NCUS',
  'NEUS',
  'NJUS',
  'NVUS',
  'NYUS',
  'OHUS',
  'ORUS',
  'PAUS',
  'TNUS',
  'TXUS',
  'USUS',
  'UTUS',
  'WAUS',
  'WIUS'],
 'MT': ['MT', 'MTMT'],
 'BR': ['BR', 'BRBR'],
 'FI': ['FI', 'FIFI'],
 'TH': ['TH', 'THTH'],
 'GH': ['GH', 'GHGH'],
 'ID': ['ID', 'IDID'],
 'CR': ['CR', 'CRCR'],
 'CA': ['CA', 'ONCA', 'CACA'],
 'SG': ['SG', 'SGSG'],
 'NDFCountry': ['',
  'oCHCH',
  'EE',
  'AM',
  'SN',
  'LT',
  'BF',
  'KW',
  '™rCHCH',
  'MGMG',
  'KN',
  'BB',
  'PEPE',
  'AG',
  'CW',
  'lHUHU',
  'JO',
  'MV',
  'AZ',
  'LK',
  'PYPY',
  'JPJP',
  'VN',
  'TJ',
  'ML',
  'TG',
  'vHUHU',
  'PR']}
 
 
def get_day(timestamp):
    day = datetime.datetime.fromtimestamp(timestamp).strftime("%A")
    return day


def get_month(timestamp):
    month = datetime.datetime.fromtimestamp(timestamp).strftime("%b")
    return month


def get_year(timestamp):
    year = datetime.datetime.fromtimestamp(timestamp).strftime("%Y")
    return year


def get_hour(timestamp):
    hour = datetime.datetime.fromtimestamp(timestamp).strftime('%H')
    return hour


def get_country(local):
    country = local[36:].strip()
    return country 


def get_city(local):
    city = local[23:36].strip()
    return city


def get_inst(local):
    inst = local[0:22].strip()
    return inst


def get_normal_country(country):
    for key in country_dict.keys():
        if country in country_dict[key]:
            return key


def get_cities_dict(FILE_PATH, CITIES_FILE_NAME):
    cities_dict = dict()
    with open(FILE_PATH + '/' + CITIES_FILE_NAME,'r') as data:
        reader = csv.reader(data)
        for line in reader:
            line[:] = [item for item in line if item != '']
            cities_dict[line[0]] = sorted(list(set(line)))
    return cities_dict


def get_normal_city(city):
    for key in cities_dict.keys():
        #print(key, city)
        for city_d in cities_dict[key]:
            #print(city, city_d)
            if city == city_d:              
                return key
                break
    else:
        return 'no city'


def linear_SVC_tuning(X_train, y_train):

    param_grid = [
        {'penalty':['l1'],
        'class_weight':[None, 'balanced'],
        'fit_intercept':[False, True],
        'dual':[False],
        'max_iter': [1000, 2000, 5000, 10000],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'loss':['squared_hinge']
        },
        {'penalty':['l2'],
        'class_weight':[None, 'balanced'],
        'fit_intercept':[False, True],
        'dual':[True, False],
        'max_iter': [1000, 2000, 5000, 10000],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
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
        {'n_estimators':[100, 200, 300, 400, 500, 1000, 2000],
        'max_depth':[1, 2, 3, 4, 5, 10, 20],
        'bootstrap':[True, False],
        'max_features':['auto', 'sqrt'],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5]
        }
    ]
    
    forest_class = RandomForestClassifier()
    
    grid_search = GridSearchCV(forest_class, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
    
    grid_search.fit(X_train_full, y_train_full)
    
    forest_cls = grid_search.best_estimator_
    
    print('RandomForest best params', '\n', grid_search.best_params_)
    print('RandomForest best score', '\n', grid_search.best_score_)
    
#    forest_cv_res = grid_search.cv_results_
#    
#    for mean_score, params in zip(forest_cv_res['mean_test_score'], forest_cv_res['params']):
#        print(mean_score, params)
        
    return forest_cls



#def feature_creator(users_list, ccy_tranccy_pairs):
#    result_dict = dict([(key,{}) for key in ccy_tranccy_pairs])
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
#    res = dict((key,value) for key, value in result_dict.items() if len(result_dict[key]) != 0)
#    return res


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
                            result_dict[pair][unique_mcc][1][0] += sum_by_mcc_user
                            result_dict[pair][unique_mcc][1][1] += 1
                        else:
                            result_dict[pair][unique_mcc][0][0] += sum_by_mcc_user
                            result_dict[pair][unique_mcc][0][1] += 1
                    else:
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair].update({unique_mcc:{1: [sum_by_mcc_user, 1], 0:[0,0]}})
                        else:
                            result_dict[pair].update({unique_mcc:{1:[0,0], 0: [sum_by_mcc_user, 1]}})
    res = dict((key,value) for key, value in result_dict.items() if len(result_dict[key]) != 0)
    return res


def day_feature_creator(users_list, ccy_tranccy_pairs):
    result_dict = dict([(key,{}) for key in ccy_tranccy_pairs])
    for user in users_list:
        for pair in ccy_tranccy_pairs:
            user_selected = user.query('ccy ==' + str(pair[0]) + ' & ' + 'tranccy == ' + str(pair[1]))
            for day in list(user_selected.day.unique()):
                sum_by_day_user = user_selected[user_selected['day'] == day]['amount'].sum()
                
                if sum_by_day_user > 0:
                
                    if day in result_dict[pair].keys():
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair][day][1][0] += sum_by_day_user
                            result_dict[pair][day][1][1] += 1
                        else:
                            result_dict[pair][day][0][0] += sum_by_day_user
                            result_dict[pair][day][0][1] += 1
                    else:
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair].update({day:{1: [sum_by_day_user, 1], 0:[0,0]}})
                        else:
                            result_dict[pair].update({day:{1:[0,0], 0: [sum_by_day_user, 1]}})
    res = dict((key,value) for key, value in result_dict.items() if len(result_dict[key]) != 0)
    return res


def hour_feature_creator(users_list, ccy_tranccy_pairs):
    result_dict = dict([(key,{}) for key in ccy_tranccy_pairs])
    for user in users_list:
        for pair in ccy_tranccy_pairs:
            user_selected = user.query('ccy ==' + str(pair[0]) + ' & ' + 'tranccy == ' + str(pair[1]))
            for hour in list(user_selected.hour.unique()):
                sum_by_hour_user = user_selected[user_selected['hour'] == hour]['amount'].sum()
                
                if sum_by_hour_user > 0:
                
                    if hour in result_dict[pair].keys():
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair][hour][1][0] += sum_by_hour_user
                            result_dict[pair][hour][1][1] += 1
                        else:
                            result_dict[pair][hour][0][0] += sum_by_hour_user
                            result_dict[pair][hour][0][1] += 1
                    else:
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair].update({hour:{1: [sum_by_hour_user, 1], 0:[0,0]}})
                        else:
                            result_dict[pair].update({hour:{1:[0,0], 0: [sum_by_hour_user, 1]}})
    res = dict((key,value) for key, value in result_dict.items() if len(result_dict[key]) != 0)
    return res


def month_feature_creator(users_list, ccy_tranccy_pairs):
    result_dict = dict([(key,{}) for key in ccy_tranccy_pairs])
    for user in users_list:
        for pair in ccy_tranccy_pairs:
            user_selected = user.query('ccy ==' + str(pair[0]) + ' & ' + 'tranccy == ' + str(pair[1]))
            for month in list(user_selected.month.unique()):
                sum_by_month_user = user_selected[user_selected['month'] == month]['amount'].sum()
                
                if sum_by_month_user > 0:
                
                    if month in result_dict[pair].keys():
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair][month][1][0] += sum_by_month_user
                            result_dict[pair][month][1][1] += 1
                        else:
                            result_dict[pair][month][0][0] += sum_by_month_user
                            result_dict[pair][month][0][1] += 1
                    else:
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair].update({month:{1: [sum_by_month_user, 1], 0:[0,0]}})
                        else:
                            result_dict[pair].update({month:{1:[0,0], 0: [sum_by_month_user, 1]}})
    res = dict((key,value) for key, value in result_dict.items() if len(result_dict[key]) != 0)
    return res


def year_feature_creator(users_list, ccy_tranccy_pairs):
    result_dict = dict([(key,{}) for key in ccy_tranccy_pairs])
    for user in users_list:
        for pair in ccy_tranccy_pairs:
            user_selected = user.query('ccy ==' + str(pair[0]) + ' & ' + 'tranccy == ' + str(pair[1]))
            for year in list(user_selected.year.unique()):
                sum_by_year_user = user_selected[user_selected['year'] == year]['amount'].sum()
                
                if sum_by_year_user > 0:
                
                    if year in result_dict[pair].keys():
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair][year][1][0] += sum_by_year_user
                            result_dict[pair][year][1][1] += 1
                        else:
                            result_dict[pair][year][0][0] += sum_by_year_user
                            result_dict[pair][year][0][1] += 1
                    else:
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair].update({year:{1: [sum_by_year_user, 1], 0:[0,0]}})
                        else:
                            result_dict[pair].update({year:{1:[0,0], 0: [sum_by_year_user, 1]}})
    res = dict((key,value) for key, value in result_dict.items() if len(result_dict[key]) != 0)
    return res


def country_feature_creator(users_list, ccy_tranccy_pairs):
    result_dict = dict([(key,{}) for key in ccy_tranccy_pairs])
    for user in users_list:
        for pair in ccy_tranccy_pairs:
            user_selected = user.query('ccy ==' + str(pair[0]) + ' & ' + 'tranccy == ' + str(pair[1]))
            for country in list(user_selected.country.unique()):
                sum_by_country_user = user_selected[user_selected['country'] == country]['amount'].sum()
                
                if sum_by_country_user > 0:
                
                    if country in result_dict[pair].keys():
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair][country][1][0] += sum_by_country_user
                            result_dict[pair][country][1][1] += 1
                        else:
                            result_dict[pair][country][0][0] += sum_by_country_user
                            result_dict[pair][country][0][1] += 1
                    else:
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair].update({country:{1: [sum_by_country_user, 1], 0:[0,0]}})
                        else:
                            result_dict[pair].update({country:{1:[0,0], 0: [sum_by_country_user, 1]}})
    res = dict((key,value) for key, value in result_dict.items() if len(result_dict[key]) != 0)
    return res


def city_feature_creator(users_list, ccy_tranccy_pairs):
    result_dict = dict([(key,{}) for key in ccy_tranccy_pairs])
    for user in users_list:
        for pair in ccy_tranccy_pairs:
            user_selected = user.query('ccy ==' + str(pair[0]) + ' & ' + 'tranccy == ' + str(pair[1]))
            for city in list(user_selected.city.unique()):
                sum_by_city_user = user_selected[user_selected['city'] == city]['amount'].sum()
                
                if sum_by_city_user > 0:
                
                    if city in result_dict[pair].keys():
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair][city][1][0] += sum_by_city_user
                            result_dict[pair][city][1][1] += 1
                        else:
                            result_dict[pair][city][0][0] += sum_by_city_user
                            result_dict[pair][city][0][1] += 1
                    else:
                        if user_selected.sexid.unique()[0] == 1:
                            result_dict[pair].update({city:{1: [sum_by_city_user, 1], 0:[0,0]}})
                        else:
                            result_dict[pair].update({city:{1:[0,0], 0: [sum_by_city_user, 1]}})
    res = dict((key,value) for key, value in result_dict.items() if len(result_dict[key]) != 0)
    return res


def threshold_counter_1(result_dict):
    count_list = [(key[0], key[1], key_0, value_0[1][0], value_0[1][1]) for key, value in result_dict.items() for key_0, value_0 in value.items() if value_0[1][0] != 0 and value_0[1][1] != 0]
    activities_counts = dict([((str(row[0]) + '_' + str(row[1]) + '_' + str(row[2])), row[4]) for row in count_list])
    temp_frame = pd.DataFrame.from_dict(activities_counts, orient='index')
    return temp_frame.quantile(q=0.9)


def threshold_counter_0(result_dict):
    count_list = [(key[0], key[1], key_0, value_0[0][0], value_0[0][1]) for key, value in result_dict.items() for key_0, value_0 in value.items() if value_0[1][0] != 0 and value_0[1][1] != 0]
    activities_counts = dict([((str(row[0]) + '_' + str(row[1]) + '_' + str(row[2])), row[4]) for row in count_list])
    temp_frame = pd.DataFrame.from_dict(activities_counts, orient='index')
    return temp_frame.quantile(q=0.9)


def fullfill_frame(columns_list, users_list, total_columns_list):
    train_data_df = pd.DataFrame(columns=columns_list, index=range(len(users_list)))
    train_data_df['sexid'] = 0
    
    for I in range(0,len(users_list)):
        for U in total_columns_list:
            t = users_list[I].query('ccy ==' + str(U[0]) + ' & ' + 'tranccy ==' + str(U[1]))
            if t[t['mcc'] == U[2]]['amount'].sum():
                train_data_df.loc[I, (str(U[0]) + '_' + str(U[1])+ '_' + str(U[2]))] = t[t['mcc'] == U[2]]['amount'].sum()
#                train_data_df[(str(U[0]) + '_' + str(U[1])+ '_' + str(U[2]))][I] = t[t['mcc'] == U[2]]['amount'].sum()
#        train_data_df['sexid'][I] = users_list[I].sexid.unique()[0]
        train_data_df.loc[I, 'sexid'] = users_list[I].sexid.unique()[0]
        
    train_data_df.fillna(0, inplace=True)
    return train_data_df


def day_fullfill_frame(train_data_df, columns_list, users_list, total_columns_list):
    
    for I in range(0,len(users_list)):
        for U in total_columns_list:
            t = users_list[I].query('ccy ==' + str(U[0]) + ' & ' + 'tranccy ==' + str(U[1]))
            if t[t['day'] == U[2]]['amount'].sum():
                train_data_df.loc[I, (str(U[0]) + '_' + str(U[1])+ '_' + str(U[2]))] = t[t['day'] == U[2]]['amount'].sum()
        
    train_data_df.fillna(0, inplace=True)
    return train_data_df
    

def hour_fullfill_frame(train_data_df, columns_list, users_list, total_columns_list):
    
    for I in range(0,len(users_list)):
        for U in total_columns_list:
            t = users_list[I].query('ccy ==' + str(U[0]) + ' & ' + 'tranccy ==' + str(U[1]))
            if t[t['hour'] == U[2]]['amount'].sum():
                train_data_df.loc[I, (str(U[0]) + '_' + str(U[1])+ '_' + str(U[2]))] = t[t['hour'] == U[2]]['amount'].sum()
        
    train_data_df.fillna(0, inplace=True)
    return train_data_df


def month_fullfill_frame(train_data_df, columns_list, users_list, total_columns_list):
    
    for I in range(0,len(users_list)):
        for U in total_columns_list:
            t = users_list[I].query('ccy ==' + str(U[0]) + ' & ' + 'tranccy ==' + str(U[1]))
            if t[t['month'] == U[2]]['amount'].sum():
                train_data_df.loc[I, (str(U[0]) + '_' + str(U[1])+ '_' + str(U[2]))] = t[t['month'] == U[2]]['amount'].sum()
        
    train_data_df.fillna(0, inplace=True)
    return train_data_df


def year_fullfill_frame(train_data_df, columns_list, users_list, total_columns_list):
    
    for I in range(0,len(users_list)):
        for U in total_columns_list:
            t = users_list[I].query('ccy ==' + str(U[0]) + ' & ' + 'tranccy ==' + str(U[1]))
            if t[t['year'] == U[2]]['amount'].sum():
                train_data_df.loc[I, (str(U[0]) + '_' + str(U[1])+ '_' + str(U[2]))] = t[t['year'] == U[2]]['amount'].sum()
        
    train_data_df.fillna(0, inplace=True)
    return train_data_df


def country_fullfill_frame(train_data_df, columns_list, users_list, total_columns_list):
    
    for I in range(0,len(users_list)):
        for U in total_columns_list:
            t = users_list[I].query('ccy ==' + str(U[0]) + ' & ' + 'tranccy ==' + str(U[1]))
            if t[t['country'] == U[2]]['amount'].sum():
                train_data_df.loc[I, (str(U[0]) + '_' + str(U[1])+ '_' + str(U[2]))] = t[t['country'] == U[2]]['amount'].sum()
        
    train_data_df.fillna(0, inplace=True)
    return train_data_df


def city_fullfill_frame(train_data_df, columns_list, users_list, total_columns_list):
    
    for I in range(0,len(users_list)):
        for U in total_columns_list:
            t = users_list[I].query('ccy ==' + str(U[0]) + ' & ' + 'tranccy ==' + str(U[1]))
            if t[t['city'] == U[2]]['amount'].sum():
                train_data_df.loc[I, (str(U[0]) + '_' + str(U[1])+ '_' + str(U[2]))] = t[t['city'] == U[2]]['amount'].sum()
        
    train_data_df.fillna(0, inplace=True)
    return train_data_df


def draft_classifiers_evaluation(df_res, y):

    classifiers = [
    LinearSVC(),
    LinearSVR(),
    KNeighborsClassifier(3),
    SVC(probability=True),
    NuSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    BernoulliNB(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    MLPClassifier(),
    SGDClassifier(),
    LogisticRegressionCV()]

    log_cols = ["Classifier", "ROC_AUC score"]
    log = pd.DataFrame(columns=log_cols)

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

    acc_dict = {}

    for train_index, test_index in sss.split(df_res, y):
        X_train, X_test = df_res.iloc[train_index], df_res.iloc[test_index]
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
    return log


def classifiers_evaluation(df_res, y):

    classifiers = [
    LinearSVC(),
    LinearSVR(),
    KNeighborsClassifier(3),
    SVC(probability=True),
    NuSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    BernoulliNB(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    MLPClassifier(max_iter=600),
    SGDClassifier(max_iter=600),
    LogisticRegressionCV(max_iter=600)]
    
    res = list()
    
    preprocess = [
            preprocessing.QuantileTransformer(),
            preprocessing.MinMaxScaler(),
            preprocessing.Normalizer(),
            preprocessing.StandardScaler(),
            preprocessing.RobustScaler(),
            preprocessing.MaxAbsScaler()
            ]
    for processor in preprocess:
        X = processor.fit_transform(df_res)
        
        log_cols = ["Classifier", "ROC_AUC score"]
        log = pd.DataFrame(columns=log_cols)

        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    
        acc_dict = {}
    
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    
            for clf in classifiers:
                name = clf.__class__.__name__
                clf.fit(X_train, y_train)
                train_predictions = clf.predict(X_test)
    #            acc = accuracy_score(y_test, train_predictions)
                acc = roc_auc_score(y_test, train_predictions)

                if name in acc_dict:
                    acc_dict[name] += acc
                else:
                    acc_dict[name] = acc
    
        for clf in acc_dict:
            acc_dict[clf] = acc_dict[clf] / 10.0
            log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
            log = log.append(log_entry)
    
        print(processor.__class__.__name__)
        print(log)
        res.append([processor.__class__.__name__, log])

    return res
            

if __name__ == '__main__':
    
    df_train, df_test = get_categorical_data()
    
    df_train['day'] = df_train['trandatetime'].apply(lambda x: get_day(x))
    df_train['hour'] = df_train['trandatetime'].apply(lambda x: get_hour(x))
    df_train['month'] = df_train['trandatetime'].apply(lambda x: get_month(x))
    df_train['year'] = df_train['trandatetime'].apply(lambda x: get_year(x))
    df_train['country'] = df_train['location'].apply(lambda x: get_country(x))
    df_train['country'] = df_train['country'].apply(lambda x: get_normal_country(x))
    
    cities_dict = get_cities_dict(FILE_PATH, CITIES_FILE_NAME)
    
    df_train['city'] = df_train['location'].apply(lambda x: get_city(x))
    df_train['city'] = df_train['city'].apply(lambda x: get_normal_city(x))
    
    day_list = df_train['day'].unique().tolist()
    hour_list = df_train['hour'].unique().tolist()
    month_list = df_train['month'].unique().tolist()
    year_list = df_train['year'].unique().tolist()
    country_list = df_train['country'].unique().tolist()
    city_list = df_train['city'].unique().tolist()
    
    users_list = process_data(df_train)
    
    ccy_tranccy_pairs = list(itertools.product(list(df_train.ccy.unique()), list(df_train.tranccy.unique())))
    
    result_dict = feature_creator(users_list, ccy_tranccy_pairs)
    
    day_result_dict = day_feature_creator(users_list, ccy_tranccy_pairs)
    
    hour_result_dict = hour_feature_creator(users_list, ccy_tranccy_pairs)
    
    month_result_dict = month_feature_creator(users_list, ccy_tranccy_pairs)
    
    year_result_dict = year_feature_creator(users_list, ccy_tranccy_pairs)
    
    country_result_dict = country_feature_creator(users_list, ccy_tranccy_pairs)
    
    city_result_dict = city_feature_creator(users_list, ccy_tranccy_pairs)
    
    city_1_list = [(key[0], key[1], key_0, value_0[1][0]) for key, value in city_result_dict.items() for key_0, value_0 in value.items() if value_0[1][0] != 0 and value_0[1][1] > threshold_counter_1(city_result_dict)[0]]
    city_0_list = [(key[0], key[1], key_0, value_0[0][0]) for key, value in city_result_dict.items() for key_0, value_0 in value.items() if value_0[0][0] != 0 and value_0[0][1] > threshold_counter_0(city_result_dict)[0]]
    city_total_columns_list = copy.deepcopy(city_1_list)
    city_total_columns_list.extend(city_0_list)
    city_columns_list = list(set([str(I[0]) + '_' + str(I[1])+ '_' + str(I[2]) for I in city_total_columns_list]))
    
    country_1_list = [(key[0], key[1], key_0, value_0[1][0]) for key, value in country_result_dict.items() for key_0, value_0 in value.items() if value_0[1][0] != 0 and value_0[1][1] > threshold_counter_1(country_result_dict)[0]]
    country_0_list = [(key[0], key[1], key_0, value_0[0][0]) for key, value in country_result_dict.items() for key_0, value_0 in value.items() if value_0[0][0] != 0 and value_0[0][1] > threshold_counter_0(country_result_dict)[0]]
    country_total_columns_list = copy.deepcopy(country_1_list)
    country_total_columns_list.extend(country_0_list)
    country_columns_list = list(set([str(I[0]) + '_' + str(I[1])+ '_' + str(I[2]) for I in country_total_columns_list]))
    
    hour_1_list = [(key[0], key[1], key_0, value_0[1][0]) for key, value in hour_result_dict.items() for key_0, value_0 in value.items() if value_0[1][0] != 0 and value_0[1][1] > threshold_counter_1(hour_result_dict)[0]]
    hour_0_list = [(key[0], key[1], key_0, value_0[0][0]) for key, value in hour_result_dict.items() for key_0, value_0 in value.items() if value_0[0][0] != 0 and value_0[0][1] > threshold_counter_0(hour_result_dict)[0]]
    hour_total_columns_list = copy.deepcopy(hour_1_list)
    hour_total_columns_list.extend(hour_0_list)
    hour_columns_list = list(set([str(I[0]) + '_' + str(I[1])+ '_' + str(I[2]) for I in hour_total_columns_list]))
    
    day_1_list = [(key[0], key[1], key_0, value_0[1][0]) for key, value in day_result_dict.items() for key_0, value_0 in value.items() if value_0[1][0] != 0 and value_0[1][1] > threshold_counter_1(day_result_dict)[0]]
    day_0_list = [(key[0], key[1], key_0, value_0[0][0]) for key, value in day_result_dict.items() for key_0, value_0 in value.items() if value_0[0][0] != 0 and value_0[0][1] > threshold_counter_0(day_result_dict)[0]]
    day_total_columns_list = copy.deepcopy(day_1_list)
    day_total_columns_list.extend(day_0_list)
    day_columns_list = list(set([str(I[0]) + '_' + str(I[1])+ '_' + str(I[2]) for I in day_total_columns_list]))

    month_1_list = [(key[0], key[1], key_0, value_0[1][0]) for key, value in month_result_dict.items() for key_0, value_0 in value.items() if value_0[1][0] != 0 and value_0[1][1] > threshold_counter_1(month_result_dict)[0]]
    month_0_list = [(key[0], key[1], key_0, value_0[0][0]) for key, value in month_result_dict.items() for key_0, value_0 in value.items() if value_0[0][0] != 0 and value_0[0][1] > threshold_counter_0(month_result_dict)[0]]
    month_total_columns_list = copy.deepcopy(month_1_list)
    month_total_columns_list.extend(month_0_list)
    month_columns_list = list(set([str(I[0]) + '_' + str(I[1])+ '_' + str(I[2]) for I in month_total_columns_list]))
    
    year_1_list = [(key[0], key[1], key_0, value_0[1][0]) for key, value in year_result_dict.items() for key_0, value_0 in value.items() if value_0[1][0] != 0 and value_0[1][1] > threshold_counter_1(year_result_dict)[0]]
    year_0_list = [(key[0], key[1], key_0, value_0[0][0]) for key, value in year_result_dict.items() for key_0, value_0 in value.items() if value_0[0][0] != 0 and value_0[0][1] > threshold_counter_0(year_result_dict)[0]]
    year_total_columns_list = copy.deepcopy(year_1_list)
    year_total_columns_list.extend(year_0_list)
    year_columns_list = list(set([str(I[0]) + '_' + str(I[1])+ '_' + str(I[2]) for I in year_total_columns_list]))
    
    sexid_1_list = [(key[0], key[1], key_0, value_0[1][0]) for key, value in result_dict.items() for key_0, value_0 in value.items() if value_0[1][0] != 0 and value_0[1][1] > threshold_counter_1(result_dict)[0]]
    sexid_0_list = [(key[0], key[1], key_0, value_0[0][0]) for key, value in result_dict.items() for key_0, value_0 in value.items() if value_0[0][0] != 0 and value_0[0][1] > threshold_counter_0(result_dict)[0]]    
    total_columns_list = copy.deepcopy(sexid_1_list)
    total_columns_list.extend(sexid_0_list)       
    columns_list = list(set([str(I[0]) + '_' + str(I[1])+ '_' + str(I[2]) for I in total_columns_list]))
    
    train_data_df = fullfill_frame(columns_list, users_list, total_columns_list)
    
    train_data_df = day_fullfill_frame(train_data_df, day_columns_list, users_list, day_total_columns_list)
    
    train_data_df = hour_fullfill_frame(train_data_df, hour_columns_list, users_list, hour_total_columns_list)
    
    train_data_df = month_fullfill_frame(train_data_df, month_columns_list, users_list, month_total_columns_list)
    
    train_data_df = year_fullfill_frame(train_data_df, year_columns_list, users_list, year_total_columns_list)
    
    train_data_df = country_fullfill_frame(train_data_df, country_columns_list, users_list, country_total_columns_list)
    
    train_data_df = city_fullfill_frame(train_data_df, city_columns_list, users_list, city_total_columns_list)
    
    y_train_full = train_data_df['sexid']
    X_train_full = train_data_df.drop('sexid', axis=1)
    
    res_classifiers_selection = classifiers_evaluation(X_train_full, y_train_full)
    
    X_train_full = X_train_full.apply(np.log)
    X_train_full[np.isneginf(X_train_full)] = 0
    
    np_log_res_classifiers_selection = classifiers_evaluation(X_train_full, y_train_full)
    
#    scores = select_preprocess(X_train_full, y_train_full)
#    
#    def select_preprocess(df_res, y):
#        res = list()
#        preprocess = [
#                preprocessing.QuantileTransformer(),
#                preprocessing.MinMaxScaler(),
#                preprocessing.Normalizer(),
#                preprocessing.StandardScaler(),
#                preprocessing.RobustScaler(),
#                preprocessing.MaxAbsScaler()
#                ]
#        for processor in preprocess:
#            print('\n')
#            print(processor.__class__.__name__)
#            X = processor.fit_transform(df_res)
#            scores = draft_classifiers_evaluation(X, y)
#            res.append([processor.__class__.__name__, scores])
#        return res
    
