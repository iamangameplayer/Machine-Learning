from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer



cancer=load_breast_cancer()

X=cancer.data
y=cancer.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=39)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(X_train)
x_test_scaled=scaler.transform(X_test)

pca=PCA(n_components=7)
x_train_pca=pca.fit_transform(x_train_scaled)
x_test_pca=pca.transform(x_test_scaled)

lr=LogisticRegression()
lr.fit(x_train_pca,y_train)

pred=lr.predict(x_test_pca)

print(f"Accuracy Score {accuracy_score(y_test,pred)}")
print(f"Confusion Matrix: {confusion_matrix(y_test,pred)}")
print(f"Classification Report :{classification_report(y_test,pred)}")

