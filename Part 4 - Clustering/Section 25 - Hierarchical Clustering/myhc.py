# Hierarchical Clustering

#%reset -f

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('mall_customers.csv')
X = dataset.iloc[:,[3,4]].values

# Using the dendogram to find the optimal # of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram Test')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Fitting the Hierarchical Clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1], s = 100, c = 'red',label = 'Careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1], s = 100, c = 'blue',label = 'Standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1], s = 100, c = 'green',label = 'Target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1], s = 100, c = 'cyan',label = 'Careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1], s = 100, c = 'magenta',label = 'Sensible')
#plt.scatter(X[y_hc==5,0],X[y_means==5,1], s = 100, c = 'purple',label = 'Sensible')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()