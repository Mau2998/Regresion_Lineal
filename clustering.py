from sklearn.cluster import KMeans
from sklearn import datasets 
from sklearn import metrics 
from matplotlib.pyplot as plt 

iris= datasets.load_iris()


x=iris.data
y=iris.target

km=KMeans(n_clusters=3,max_iter=300)
km.fit(x)
p=km.predict(x)
print p
score =metrics.adjusted_rand_score(y,p)
print score
plt.subtrama(221)
plt.scatter(x,y)