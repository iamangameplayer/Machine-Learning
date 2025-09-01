from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

housing=fetch_california_housing()
X=housing.data
y=housing.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=37)

rf=RandomForestRegressor(
    max_depth=None,
    n_estimators=1700,
    criterion='squared_error',
    min_samples_split=5,
    min_samples_leaf=4,
    n_jobs=-1
)
rf.fit(X_train,y_train)

pred=rf.predict(X_test)

print(f"Mean Squared Error:  {mean_squared_error(y_test,pred)}")
print(f"Root Mean Squared Error : {root_mean_squared_error(y_test,pred)}")