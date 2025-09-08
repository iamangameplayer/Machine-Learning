
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score,completeness_score,v_measure_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons


X,y=make_moons(n_samples=500,random_state=39,noise=0.1)

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

dbscan=DBSCAN(eps=0.27,min_samples=6)
dbscan.fit(X_scaled)
sil=silhouette_score(X_scaled,dbscan.labels_)
com=completeness_score(y,dbscan.labels_)
v_score=v_measure_score(y,dbscan.labels_)

print(f"Silhouette Score : {sil}")
print(f"Completeness Score : {com}")
print(f"V_Measure Score : {v_score}")

