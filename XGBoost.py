from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import save_model

X,y=load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
model=xgb.XGBClassifier()
model.fit(X_train,y_train)

prediction=model.predict(X_test)
accuracy=accuracy_score(y_test,prediction)
print(f"Accuracy:{accuracy:.5f}")


