import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split 
import matplotlib.pyplot as plt

#leemos la base de datos 
b=pd.read_csv('USvideos.csv',header=0)
#codigo para graficar de manera visual la eleccion de datos 

plt.scatter(b["views"],b["likes"])
plt.xlabel("v")
plt.ylabel("l")
plt.show()

ba=pd.DataFrame(b)

x=ba["views"]
y=ba["likes"]

np.array(x)
np.array(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
x_tr= x_train.reshape(11241,1)
y_tr= y_train.reshape(11241,1)
regre= LinearRegression()
#error (shapes (1,5537) and (1,1) not aligned: 5537 (dim 1) != 1 (dim 0) )
"""
regre.fit(x_tr,y_tr)
y_pre=regre.predict(x_test)
print y_pre

plt.scatter(x_test,y_test,'black')
plt.plot(x_test,y_pre,color='blue',linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
"""
