# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 21:43:16 2018

@author: User
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
X_FILE_NAME = 'new_dataframe_1.csv'



# define baseline model
def baseline_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(32, input_dim=268, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.009)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    

#    train_data = np.genfromtxt(FILE_PATH + X_FILE_NAME, delimiter=",")[1:]
#    train_labels = np.genfromtxt(FILE_PATH + Y_FILE_NAME, delimiter=",")[1:]
    
    X_train_full = pd.read_csv(FILE_PATH + X_FILE_NAME)
    y_train = X_train_full['sexid']
    
    X_train_full.drop(['Unnamed: 0'], axis=1, inplace=True)
    X_train_full.drop(['sexid'], axis=1, inplace=True)
#    y_train.drop([0], axis=1, inplace=True)
    
#    X_train_full_norm = X_train_full.div(X_train_full.sum(axis=1), axis=0)
    
    X_train_full = X_train_full.apply(np.log)
    X_train_full[np.isneginf(X_train_full)] = 0
    
    norm = preprocessing.Normalizer()
    X_train_full_norm = norm.fit_transform(X_train_full)
    
#    dataset = X_train_full_norm.values
    dataset = X_train_full_norm
    
    X = dataset[:,0:268].astype(float)
    y = y_train.values
    
    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=10, verbose=0)
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    
    results = cross_val_score(estimator, X, y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
    model = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=10, verbose=0)
    
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X, y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    

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