from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

house=load_breast_cancer()
X=house.data
y=house.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=34)

knn_model=Pipeline([
    ('scaler',StandardScaler()),
    ('knn',KNeighborsClassifier(n_neighbors=3))
])

knn_model.fit(X_train,y_train)
pred=knn_model.predict(X_test)
print(f"Accuracy Score :{accuracy_score(y_test,pred)}")
print(f"Confusion Matrix :{confusion_matrix(y_test,pred)}")
print(f"Classification Report  : {classification_report(y_test,pred)}")
