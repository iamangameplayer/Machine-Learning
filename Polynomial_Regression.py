from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures , StandardScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error,r2_score
from sklearn.pipeline import Pipeline

house = fetch_california_housing()
X=house.data
y=house.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=45)

poly_model=Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('scaler',StandardScaler()),
    ('regressor',LinearRegression())

])

poly_model.fit(X_train,y_train)

pred=poly_model.predict(X_test)

print(f"Means Squared Error {mean_squared_error(y_test,pred)}")
print((f"Root Mean Squared Error {root_mean_squared_error(y_test,pred)}"))
print((f"R2 Score {r2_score(y_test,pred)}"))
