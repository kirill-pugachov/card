# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:33:50 2018

@author: tom
"""

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
#from keras.layers import Embedding
#from keras.layers import LSTM
from sklearn import preprocessing
import pandas as pd
import numpy as np


#FILE_PATH = 'Data/'
FILE_PATH = 'C:\\Users\\tom\\card_garbage\\Data\\'
X_FILE_NAME = 'X_train_full_2809.csv'
Y_FILE_NAME = 'y_train_2809.csv'



# define baseline model
def baseline_model(optimizer='RMSprop', drop=0.3, l2_reg=0.005, init_mode='uniform'):
    model = Sequential()
    model.add(Dense(64, input_dim=1492, kernel_initializer=init_mode, activation='relu'))
#    model.add(Dense(96, input_dim=(268,), kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(64, activation='relu', kernel_initializer=init_mode, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(drop))
    model.add(Dense(64, activation='relu', kernel_initializer=init_mode, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(drop))
    model.add(Dense(32, activation='relu', kernel_initializer=init_mode, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(drop))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init_mode))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    
   
    X_train_full = pd.read_csv(FILE_PATH + X_FILE_NAME, index_col=0)
    y_train = pd.read_csv(FILE_PATH + Y_FILE_NAME, index_col=0, header=None)
    
    dataset = X_train_full.values
    
    X_prom = dataset[:,0:1492].astype(float)
    X = X_prom / (X_prom.max() + 1)
    y = y_train.values
    
    estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=64, verbose=0)
    
    optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    dropers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    l2_regs = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
    initializers = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    
    param_grid = dict(optimizer=optimizers, drop=dropers, l2_reg=l2_regs, init_mode=initializers)
    estimator_train = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=64, verbose=0)
    grid = GridSearchCV(estimator=estimator_train, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X, y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))