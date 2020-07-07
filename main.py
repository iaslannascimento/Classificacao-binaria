#Autor Iaslan Nascimento
#classificador binário para classificação de cancer
#
import pandas as pd 

#leitura do arquivo que contém os atributos
previsores = pd.read_csv('entradas_breast.csv')
#leitura dos arquivos que contém as classes
classe = pd.read_csv('saidas_breast.csv')
#importando para separar nossa base em treino e teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe, test_size=0.25)
import keras
from keras.models import Sequential
from keras.layers import Dense

classificador = Sequential()
#adicionando o número de neuronios 
#calculo = Nº de entradas + Nº de neuronios na camada de saida e divide por 2 e aredonda pra cima 
#(30 + 1) / 2 = 15.5 = 16
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform',
                        input_dim = 30 ))
print ('ok')