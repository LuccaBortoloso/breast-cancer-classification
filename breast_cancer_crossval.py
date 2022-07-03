import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

def createNeuralNet():
    classifier = Sequential()
    classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform',
                         input_dim=30))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(learning_rate=0.001, epsilon=0.0001, clipvalue= 0.5)
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', 
                       metrics=['binary_accuracy'])
    return classifier

# Carregando as bases
previsores = pd.read_csv('datasets/entradas_breast.csv')
classe = pd.read_csv('datasets/saidas_breast.csv')

# Criando o classificador
classifier = KerasClassifier(build_fn=createNeuralNet, epochs=100, batch_size=10)

results = cross_val_score(estimator=classifier, X = previsores, y = classe,
                          cv=10, scoring='accuracy')

mean = results.mean()
std = results.std()

