# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 21:43:16 2018

@author: User
"""

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from sklearn import preprocessing
import pandas as pd
import numpy as np


FILE_PATH = 'Data/'
X_FILE_NAME = 'X_train_full_2809.csv'
Y_FILE_NAME = 'y_train_2809.csv'


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=1492, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


if __name__ == '__main__':
    

#    train_data = np.genfromtxt(FILE_PATH + X_FILE_NAME, delimiter=",")[1:]
#    train_labels = np.genfromtxt(FILE_PATH + Y_FILE_NAME, delimiter=",")[1:]
    
    X_train_full = pd.read_csv(FILE_PATH + X_FILE_NAME)
    y_train = pd.read_csv(FILE_PATH + Y_FILE_NAME, header=None)
    
    X_train_full.drop(['Unnamed: 0'], axis=1, inplace=True)
    y_train.drop([0], axis=1, inplace=True)
    
    X_train_full_norm = X_train_full.div(X_train_full.sum(axis=1), axis=0)
    
    dataset = X_train_full_norm.values
    
    X = dataset[:,0:1492].astype(float)
    y = y_train.values
    
    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=1)
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    
    results = cross_val_score(estimator, X, y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
#    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
#    X_train_min = scaler.fit_transform(X_train_full)
    
#    x_val = train_data[:1000]
#    partial_x_train = train_data[1000:]
#    
#    y_val = train_labels[:1000]
#    partial_y_train = train_labels[1000:]
#    
##    seq_length = 1493
#    max_features = 1493
    
#    from keras.models import Sequential
#    from keras.layers import Dense, Dropout
#    from keras.layers import Embedding
#    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
#    
#    seq_length = 7917
#    
#    model = Sequential()
#    model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 1493)))
#    model.add(Conv1D(64, 3, activation='relu'))
#    model.add(MaxPooling1D(3))
#    model.add(Conv1D(128, 3, activation='relu'))
#    model.add(Conv1D(128, 3, activation='relu'))
#    model.add(GlobalAveragePooling1D())
#    model.add(Dropout(0.5))
#    model.add(Dense(1, activation='sigmoid'))
#    
#    model.compile(loss='binary_crossentropy',
#                  optimizer='rmsprop',
#                  metrics=['accuracy'])
#    
#    model.fit(partial_x_train, partial_y_train, batch_size=16, epochs=10)
#    score = model.evaluate(x_val, y_val, batch_size=16)
#    
#    model = Sequential()
#    model.add(Embedding(max_features, output_dim=256))
#    model.add(LSTM(128))
#    model.add(Dropout(0.5))
#    model.add(Dense(1, activation='sigmoid'))
#    
#    model.compile(loss='binary_crossentropy',
#                  optimizer='rmsprop',
#                  metrics=['accuracy'])
#    
#    model.fit(partial_x_train, partial_y_train, batch_size=16, epochs=10)
#    score = model.evaluate(x_val, y_val, batch_size=16)