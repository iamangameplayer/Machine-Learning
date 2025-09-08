

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,completeness_score,v_measure_score
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris=load_iris()

X=iris.data
y=iris.target

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
pca=PCA(n_components=3)

X_pca=pca.fit_transform(X_scaled)

km=KMeans(n_clusters=3,random_state=45,n_init=5)
km.fit(X_pca)

print(f"Silhouette Score {silhouette_score(X_pca,km.labels_)}")
print(f"Completeness Score : {completeness_score(y,km.labels_)}")
print(f"V Measure Score : {v_measure_score(y,km.labels_)}")

