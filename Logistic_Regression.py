from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , accuracy_score


iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.6,random_state=26)

model=LogisticRegression()
model.fit(X_train,y_train)

prediction=model.predict(X_test)
print(f"Accuracy is {accuracy_score(y_test,prediction)}")
print(f"Mean Squared Error is {mean_squared_error(y_test,prediction)}")