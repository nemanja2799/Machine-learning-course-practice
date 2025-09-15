import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("part_4_clustering/Section_25_Hierarchical_Clustering/Mall_Customers.csv")
X = dataset.iloc[ : , [3, 4]].values

# implementing dendogram to visualise hierarchical clustering process and decide number of clusters
import scipy.cluster.hierarchy as sch

# first parameter is X(dataset) and second is method of clustering - ward which means minimizing the variants inside your clusters
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel("Customers")
plt.ylabel("Euclidian distances")
plt.show()

# creating hierarchical clustering model
from sklearn.cluster import AgglomerativeClustering 
# in new version affinity parameter is removed and distance metric is always euclidian

hc = AgglomerativeClustering(n_clusters= 5,  linkage = 'ward')
# we dont only train model but also create predicted data clusters with fit_predict method

y_hc = hc.fit_predict(X)
# we plot clusters on graph that contains anual income ans spending score as axis
# second coordinate is anual income and first is spending score
plt.scatter(X[y_hc==0,0], X[y_hc ==0,1], s=100, c='red', label='Cluster1')
plt.scatter(X[y_hc ==1,0], X[y_hc ==1,1], s=100, c='blue', label='Cluster2')
plt.scatter(X[y_hc ==2,0], X[y_hc ==2,1], s=100, c='green', label='Cluster3')
plt.scatter(X[y_hc ==3,0], X[y_hc ==3,1], s=100, c='cyan', label='Cluster4')
plt.scatter(X[y_hc ==4,0], X[y_hc ==4,1], s=100, c='magenta', label='Cluster5')

plt.title("Clusters of customers")
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()
