import numpy as np
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Generating Random house prices
np.random.seed(23)
X=np.random.randint(1000,2000,size=(100,1))
Y=3000*X.flatten()+np.random.randint(-30000,30000,size=100)

#Data split into train and test
X_train ,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=23)

#Training Linear Regression Model
model=LinearRegression()#Linear Regression
model.fit(X_train,Y_train)

#Prediction
prediction=model.predict(X_test)

#Evaluation
a=mean_absolute_error(Y_test,prediction)
b=mean_squared_error(Y_test,prediction)
c=np.sqrt(b)
d=r2_score(Y_test,prediction)
print(f"Mean Squared Error is {b}")
print(f"Mean Absolute Error is {a}")
print(f"Root Mean Squared Error is {c}")
print(f"R2_Score is {d}")


#Plotting results
plt.scatter(X_train,Y_train , color='orange', label='Training Data') #Training Data
plt.scatter(X_test,Y_test,color='red',label ='Actual Prices')#Original Data
plt.plot(X_test,prediction,color='black', linewidth=2)#Regression Line
plt.show()
