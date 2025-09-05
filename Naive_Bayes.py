from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

cancer=load_breast_cancer()
X=cancer.data
y=cancer.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=31)

nb=GaussianNB()
nb.fit(X_train,y_train)

pred=nb.predict(X_test)

print(f"Confusion Matrix : {confusion_matrix(y_test,pred)}")
print(f"Classification Report : {classification_report(y_test,pred)}")
print(f"Accuracy Score : {accuracy_score(y_test,pred)}")