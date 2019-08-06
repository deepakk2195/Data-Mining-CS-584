import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import mixture
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm
import seaborn as sns
from scipy.spatial.distance import cdist


X = pd.read_csv('dataset1.csv', header = None)
print(X.shape)
pdist=cdist(X,X,metric="cosine")
print(pdist)
Y=X
K=X
#kmeans
sse = {}
for k in range(2,5):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
kmeans = KMeans(n_clusters=3).fit(X)
centroids = kmeans.cluster_centers_
label=kmeans.labels_
sil_coeff = silhouette_score(X, label, metric='euclidean')
print("silhouette score:"+str(sil_coeff))
g=sns.clustermap(X, metric="correlation")
g=sns.clustermap(pdist, metric="correlation")
plt.show()

#dbscan
scaler = StandardScaler()
X_scaled = scaler.fit_transform(Y)
db = DBSCAN(eps=0.50, min_samples=10)
clusters = db.fit_predict(X_scaled)
label=db.labels_
sil_coeff = silhouette_score(X_scaled, label, metric='euclidean')
print("silhouette score:"+str(sil_coeff))
X_scaled=pd.DataFrame(X_scaled)
g=sns.clustermap(X_scaled, metric="correlation")
plt.show()

#EM
clf = mixture.BayesianGaussianMixture(n_components=7, covariance_type='diag', n_init=5, max_iter=1000)
clf.fit(K)
labels=clf.predict(K)
sil_coeff =silhouette_score(K, labels)
print("silhouette score:"+str(sil_coeff))
g=sns.clustermap(K, metric="correlation")
plt.show()




#dataset2
X1 = pd.read_csv('dataset2.csv', header = None)
print(X1.shape)
pdist=cdist(X1,X1,metric="cosine")
print(pdist)

Y=X1
K=X1
#kmeans
sse = {}
for k in range(2,5):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X1)
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
kmeans = KMeans(n_clusters=3).fit(X1)
centroids = kmeans.cluster_centers_
label=kmeans.labels_
sil_coeff = silhouette_score(X1, label, metric='euclidean')
print("silhouette score:"+str(sil_coeff))
g=sns.clustermap(X1, metric="correlation")
g=sns.clustermap(pdist, metric="correlation")
plt.show()

##dbscan
scaler = StandardScaler()
X_scaled = scaler.fit_transform(Y)
db = DBSCAN(eps=0.5, min_samples=3)
clusters = db.fit_predict(X_scaled)
label=db.labels_
sil_coeff = silhouette_score(X_scaled, label, metric='euclidean')
print("silhouette score:"+str(sil_coeff))
X_scaled=pd.DataFrame(X_scaled)
g=sns.clustermap(X_scaled, metric="correlation")
plt.show()

#EM
clf = mixture.BayesianGaussianMixture(n_components=3, covariance_type='diag', n_init=5, max_iter=1000)
clf.fit(K)
labels=clf.predict(K)
sil_coeff =silhouette_score(K, labels)
print("silhouette score:"+str(sil_coeff))
g=sns.clustermap(K,metric="correlation")
plt.show()


