# Importações
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Carregando as bases
previsores = pd.read_csv('datasets/entradas_breast.csv')
classe = pd.read_csv('datasets/saidas_breast.csv')

# Divisão do treino e teste
X_train, X_test, y_train, y_test = train_test_split(previsores, classe, test_size = 0.25)

# Criando a estrutura da rede neural
classifier = Sequential()
classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform',
                     input_dim=30)) # units = (n° entradas + n° neuronios saída) / 2
classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
classifier.add(Dense(units=1, activation='sigmoid'))

optimizer = keras.optimizers.Adam(learning_rate=0.001, epsilon=0.0001, clipvalue= 0.5)
classifier.compile(optimizer=optimizer, loss='binary_crossentropy', 
                   metrics=['binary_accuracy'])

'''
Podemos usar o otimizador de forma padrão usando:
    classifier.compile(optimizer='adam', loss='binary_crossentropy', 
                   metrics=['binary_accuracy'])
'''

weight0 = classifier.layers[0].get_weights()
print(weight0)
print(len(weight0))

weight1 = classifier.layers[1].get_weights()
weight2 = classifier.layers[2].get_weights()

# Treinando e testando o modelo
classifier.fit(X_train, y_train, batch_size=10, epochs=100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Resultado das previsões
precision = accuracy_score(y_test, y_pred)
cf_matrix = confusion_matrix(y_test, y_pred)
result = classifier.evaluate(X_test, y_test)