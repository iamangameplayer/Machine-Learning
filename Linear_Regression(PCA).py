from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine=pd.read_csv(url,sep=";")
X=wine.drop("quality",axis=1)
y=wine["quality"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=39)

scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
pca=PCA(n_components=5)
X_test_pca=pca.fit_transform(X_test_scaled)
X_train_pca=pca.transform(X_train_scaled)

reg=LinearRegression()
reg.fit(X_train_pca,y_train)

pred=reg.predict(X_test_pca)

print(f"R2 Score  {r2_score(y_test,pred)}")
print(f"Mean Square Root  {mean_squared_error(y_test,pred)}")
print(f"Root Mean Square Root  {root_mean_squared_error(y_test,pred)}")
