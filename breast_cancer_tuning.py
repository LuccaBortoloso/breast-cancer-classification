import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

def createNeuralNet(optimizer, loss, kernel_initializer, activation, neurons):
    classifier = Sequential()
    classifier.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer,
                         input_dim=30))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss=loss, 
                       metrics=['binary_accuracy'])
    return classifier

# Carregando as bases
previsores = pd.read_csv('datasets/entradas_breast.csv')
classe = pd.read_csv('datasets/saidas_breast.csv')

# Criando o classificador
classifier = KerasClassifier(build_fn=createNeuralNet)

parameters = {
        'batch_size': [10,30],
        'epochs': [50, 100],
        'optimizer': ['adam', 'sgd'],
        'loss': ['binary_crossentropy', 'hinge'],
        'kernel_initializer': ['random_uniform', 'normal'],
        'activation': ['relu', 'tanh'],
        'neurons': [8, 16, 32]
    }

grid_search = GridSearchCV(estimator= classifier, param_grid= parameters,
                           scoring= 'accuracy', cv=5)

grid_search = grid_search.fit(previsores, classe)
best_parameters = grid_search.best_params_
best_precision = grid_search.best_score_