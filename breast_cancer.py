import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json

# Carregando as bases
previsores = pd.read_csv('datasets/entradas_breast.csv')
classe = pd.read_csv('datasets/saidas_breast.csv')


# Criando o classificador
classifier = Sequential()
classifier.add(Dense(units=8, activation='relu', kernel_initializer='random_uniform',
                     input_dim=30))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=8, activation='relu', kernel_initializer='random_uniform'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', 
                   metrics=['binary_accuracy'])
classifier.fit(previsores, classe, batch_size=10, epochs=100)


# Salvando o modelo
classifier_json = classifier.to_json()

with open('breast_classifier.json', 'w') as json_file:
    json_file.write(classifier_json)
    
classifier.save_weights('breast_classifier.h5')


# Carregando o modelo
model_json = open('classifier_breast.json', 'r')
model = model_json.read()
model_json.close()

classifier = model_from_json(model)
classifier.load_weights('breast_classifier.h5')