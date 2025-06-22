import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("part_2_regresion/Section_4_Simple_Linear_Regression/Python/Salary_Data.csv")
X = dataset.iloc[: , : -1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# import desirable library for model then create object instance from class of model we want
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# train model 
regressor.fit(X_train, y_train)

# use trained model to predict result on test set 
y_test_predicted = regressor.predict(X_test)

#print(y_test)
#print(y_test_predicted)

# plot points on graph with x and y pair cordinates
plt.scatter(X_train, y_train, color='red')
#plot line of prediction, second argumet is prediction on train model 
# which we didn't have calculated in variable yet
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, regressor.predict(X_train))

plt.title('Salary vs Expiriance(Training set)')
plt.xlabel('Years of expiriance')
plt.ylabel('Salary')
plt.show()

# # plot points on graph with x and y pair cordinates
# plt.scatter(X_test, y_test, color='blue')
# #plot line of prediction, second argumet is prediction on train model 
# # which we didn't have calculated in variable yet
# plt.plot(X_train, regressor.predict(X_train))

# plt.title('Salary vs Expiriance(Test set)')
# plt.xlabel('Years of expiriance')
# plt.ylabel('Salary')
# plt.show()