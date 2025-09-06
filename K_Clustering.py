from sklearn.cluster import KMeans
from sklearn.metrics import (
silhouette_score,
completeness_score,
v_measure_score,
confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris=load_iris()
X=iris.data
y=iris.target

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

km=KMeans(n_clusters=3)
km.fit(X_scaled)

sil=silhouette_score(X_scaled,km.labels_)

comp=completeness_score(y,km.labels_)
v=v_measure_score(y,km.labels_)

print(f"Silhouette Score : {sil}")
print(f"Completeness Score : {comp}")
print(f"V_Measure Score : {v}")
print(f"Confusion Matrix : {confusion_matrix(y,km.labels_)}")
