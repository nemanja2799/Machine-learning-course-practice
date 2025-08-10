import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#for learning purpose we will keep just 2 columns anual income and spending score
# also first column ID is irrelevant for 

dataset = pd.read_csv("part_4_clustering/Section_24_K_Means_Clustering/Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

# there is no dependent variable y(we create it in clustering)
# there is no split in training and test set because there is no dependent variable

# implement elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
# we will call K Means alghorithm for 1 to 10 numbers of clusters and compute WCSS 
# so we need list to put this results

wcss = []

# init - paramether to set kmeans ++ method to avoid random initialization trap
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    # way to calculate wcss is inertia_ method
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("The elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
# from plot we see ideal number of clusters is 5

kmeans = KMeans(n_clusters = 5, init='k-means++', random_state=42)
# fit_predict not only create model and train it, but also
# return dependent variable as 1,2,3,4,5 values for each cluster

y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)
# we plot clusters on graph that contains anual income ans spending score as axis
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100, c='red', label='Cluster1')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100, c='blue', label='Cluster2')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100, c='green', label='Cluster3')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100, c='cyan', label='Cluster4')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=100, c='magenta', label='Cluster5')
#plot centroids - method gets 2 d array of coordinates of centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300,c = 'yellow', label = 'Centroids')
plt.title("Clusters of customers")
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()
