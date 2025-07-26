import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("part_2_regresion/Section 7_Support_Vector_Regression_(SVR)/Python/Position_Salaries.csv")
X = dataset.iloc[ : , 1 : -1].values
y = dataset.iloc[ : , -1].values

print(X)
print(y)

# we need to change format of y to 2D array so it will be fit for feature scaling
# it also can be done y.reshape(len(y),1)
y = np.reshape(y, (len(y), 1))
print(y)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

print(X)
print(y)
# svm module contai ns SVR class for support vector regressin

from sklearn.svm import SVR

# this model need to give input parameter of kernel which will be explained later in course and then 
# i will give more coment
# rbf kernel is most used kernel
regressor = SVR(kernel = "rbf")
regressor.fit(X, y)

# for prediction we also need to applay feature scaling on input and we will get feature scaled
# result 
# just for SVR we need to applay reshape to avoid format error
y_pred = regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1)

y_pred_transformed = sc_y.inverse_transform(y_pred)
print(y_pred_transformed)

# plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
# plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color= "blue")
# plt.title("Suport Vector Regression")
# plt.xlabel("Position level")
# plt.ylabel("Salary")

# plt.show()

# beautiful print(smooth print)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color='green')
plt.title("Truth or Bluff (Linear Reggresion) VS Polynomminal Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")

plt.show()