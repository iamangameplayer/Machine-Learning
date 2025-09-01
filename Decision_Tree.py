from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import RandomizedSearchCV

house=fetch_california_housing()
X=house.data
y=house.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=39)

model=DecisionTreeRegressor(random_state=41)


param_dist={
    'max_depth':[6,10,11,8,9,7,15,None],
    'min_samples_split':[2,10,20,40,50,100,200,225,250],
    'min_samples_leaf':[1,3,5,9,12,15,20,25,30,35,40,50]
}

rand_search=RandomizedSearchCV(estimator=model,cv=5,param_distributions=param_dist,n_iter=30,random_state=38,scoring='neg_mean_squared_error',n_jobs=-1)
rand_search.fit(X_train,y_train)
print(f"Best Parameters:{rand_search.best_params_}")

best_model=rand_search.best_estimator_
pred=best_model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test,pred)}")
print(f"Root Mean Squared Error : {root_mean_squared_error(y_test,pred)}")
