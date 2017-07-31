# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn 

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Iniciar Datos de vivienda de Boston 
bost =load_boston()

#añadir precios de vivienda
bos=pd.DataFrame(bost.data)
bos.columns=bost.feature_names
bost.target[:50]
bos['PRICE']=bost.target

#Usar solo una caracteristica

"""
y=precios de vivienda
x=variables independientes
"""

x = bos.drop('PRICE',axis=1)
#Dividir los datos en conjuntos de entrenamiento /
boston_x_train,boston_x_test,boston_y_train,boston_y_test =sklearn.model_selection.train_test_split(x,bos.PRICE,test_size=0.33,random_state=5)

#crear regresion linear 
regre = LinearRegression()

#entrenar el modelo usando el conjunto de entrenamiento 
regre.fit(boston_x_train,boston_y_train)


#el coeficiente
print 'Coeficientes :',regre.coef_

#el error cuadratico medio 
print("El error es :%.2f"
      % np.mean((regre.predict(boston_x_test) - boston_y_test) ** 2))
      
#explicacion de cantidad de variacion 
print ('Cantidad de variacion: %.2f' % regre.score(boston_x_test,boston_y_test))
#Prediccion de precios
preciosPre=regre.predict(x)
print preciosPre

#Dibujado de Graficas
plt.scatter(bos.PRICE, preciosPre, color='blue')

plt.xlabel("Precios: $y_i$")
plt.ylabel("Precios Predecidos: $\hat{y}_i$")
plt.title("Precios Vs Precios predecidos: $y_i$ vs $\hat{y}_i$ ")

plt.show()
#Regresion logistica
print "El promedio es:",np.mean(preciosPre)
list = []
variable_objetivo = preciosPre
for i in variable_objetivo:
    if i>np.mean(variable_objetivo): list.append(1)
    else: list.append(0)
np.array(list)
bos['CLASIFICADORES']=list
print bos
