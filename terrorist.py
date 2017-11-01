# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split


baseDeDatosTe=pd.read_csv('globalterrorismdb_0617dist.csv',header=0)



ter=pd.DataFrame(baseDeDatosTe)
x=ter['region']
y=ter['iyear']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=42)

X_train = tuple(range(127762))
X_train = np.asarray(X_train)

#same for y_train
Y_train=tuple(range(127762))
Y_train = np.asarray(Y_train)

#convert tuples to nparray and reshape the x_train
X_train = X_train.reshape(127762,1)

#check if shape if (127762,)
print X_train.shape
print Y_train.shape

regre=LinearRegression()

regre.fit(X_train,Y_train)

#el coeficiente
print 'Coeficientes :',regre.coef_

#explicacion de cantidad de variacion 
print ('Cantidad de variacion: %.2f' % regre.score(X_test,Y_test))



