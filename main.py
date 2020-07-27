#Autor Iaslan Nascimento
#classificador binário para classificação de cancer
#
import pandas as pd 
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

#leitura do arquivo que contém os atributos
previsores = pd.read_csv('entradas_breast.csv')
#leitura dos arquivos que contém as classes
classe = pd.read_csv('saidas_breast.csv')
#importando para separar nossa base em treino e teste

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe, test_size=0.25)
classificador = Sequential()
#adicionando o número de neuronios 
#calculo = Nº de entradas + Nº de neuronios na camada de saida e divide por 2 e aredonda pra cima 
#(30 + 1) / 2 = 15.5 = 16
#camada de entrada
#função relu pq existe uma conexão entre todos os neuronios
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform',
                        input_dim = 30 ))
#camada de saida
#função sigmoid pq a saida é binária 
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, 
                  batch_size = 10, epochs = 100)

previsoes = classificador.predict(previsores_teste)
print(previsoes)
print ('ok')