# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17

@author: kvms
"""

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

from pandas.plotting import scatter_matrix

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score

from sklearn.cluster import AgglomerativeClustering
from collections import Counter

#%%
# **************************
# *** Leitura do dataset ***
# **************************

dataset = pd.read_csv("Sales_Transactions_Dataset_Weekly.csv", index_col=0)

atributosNormalizados = dataset.columns[54:]

# Fazendo uma cópia do dataset somente com os atributos normalizados
datasetNorm = dataset[atributosNormalizados].copy()

#print("Número de colunas:", len(columns))
#print(columns)

print(datasetNorm.head(5))

X = datasetNorm.values

print("Dimensões de X:", np.shape(X) )

#%%
# *************************************
# *** Análise Exploratória de Dados ***
# *************************************

print("Criando histogramas dos dados por classes")
datasetNorm.hist()
plt.show()

print("Criando gráficos de dispersão dos dados")
scatter_matrix(datasetNorm)
plt.show()

#%%
# *************************************
# *** Principal Components Analysis ***
# *************************************

pca3 = PCA(n_components=3)
XPCA = pca3.fit_transform(X)

print("Valores de X PCA(n=3): ", XPCA.shape)

#%%
# ********************
# *** Kmeans Elbow ***
# ********************
# The K-means algorithm aims to choose centroids that minimise the inertia, 
# or within-cluster sum-of-squares criterion:

wcss1 = []
wcss2 = []
wcss3 = []

maxit = 30

for i in range(1, maxit):
    kmeans1 = KMeans(n_clusters = i)
    kmeans2 = KMeans(n_clusters = i, init = 'random')
    kmeans3 = KMeans(n_clusters = i, init = 'random', max_iter=1000)
    kmeans1.fit(X)
    kmeans2.fit(X)
    kmeans3.fit(X)
    print(i, kmeans1.inertia_, kmeans2.inertia_, kmeans3.inertia_)
    wcss1.append(kmeans1.inertia_)
    wcss2.append(kmeans2.inertia_)
    wcss3.append(kmeans3.inertia_)
plt.plot(range(1, maxit), wcss1)
plt.plot(range(1, maxit), wcss2)
plt.plot(range(1, maxit), wcss3)
plt.title('O Método Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()

#%%
# ************************
# *** Aplicando KMeans ***
# ************************

valoresK = [5, 7, 12, 15, 20, 25, 30]

kClf = [ KMeans(n_clusters=i, init='random', 
             random_state=1, max_iter=1000) for i in valoresK]

kClf = [kClf[i].fit(X) for i in range(len(kClf))]

kClfPCA = [ KMeans(n_clusters=i, init='random', 
             random_state=1, max_iter=1000) for i in valoresK]

kClfPCA = [kClfPCA[i].fit(XPCA) for i in range(len(kClfPCA))]

# Fixando o número de clusters
C = kClf[0].cluster_centers_
CPCA = kClfPCA[0].cluster_centers_

print(CPCA.shape)

#%%
# ***************************
# *** Plotagem de gráfico ***
# ***************************

print("Centroides com k=20")

fig = plt.figure()
ax = Axes3D(fig)
#ax.scatter(X[:, a1], X[:, a2], X[:, a3])
#ax.scatter(C[:, a1], C[:, a2], C[:, a3], marker='*', c='#050505', s=1000)

ax.scatter(XPCA[:, 0], XPCA[:, 1], XPCA[:, 2])           
ax.scatter(CPCA[:, 0], CPCA[:, 1], CPCA[:, 2], 
           marker='*', c='#050505', s=1000)

plt.show()

#%%
# ********************************
# *** Agrupamento aglomerativo ***
# ********************************

n_clusters = 20

# criamos o objeto para realizar o agrupamento aglomerativo
Hclustering = AgglomerativeClustering(n_clusters=n_clusters,
                                      affinity="euclidean", linkage="ward")
Hclustering.fit(X)

# Getting the cluster labels
labels = Hclustering.fit_predict(X)

silhouette_avg = silhouette_score(X, labels)
print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

print(Counter(labels).keys())
print(Counter(labels).values())
#print(labels)