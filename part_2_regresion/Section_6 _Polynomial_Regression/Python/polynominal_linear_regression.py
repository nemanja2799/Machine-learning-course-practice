import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv("part_2_regresion/Section_6 _Polynomial_Regression/Python/Position_Salaries.csv")
# here in dataset position is alredy encoded in level in second column so first 
# column is not nedded in our model

X = dataset.iloc[ : , 1 : -1].values
y = dataset.iloc[ : , -1].values

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, y)


#idea is to transform data in polinominal dataset(bo + b1*x^1 + b2*x^2 + ... + bn * x^n)
from sklearn.preprocessing import PolynomialFeatures

# degree tells us to what degree we want to expand dataset equasion
# with increase of degree we get better result for our model
poly_reg = PolynomialFeatures(degree = 4)

#then we apply this model to train poly_reg and then to use this
#poly_reg to transform X
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)

# for more beautiful and smoother visualisation 
# we add decimal points between numbwers for examle 1 and 2:  1.1, 1.2 ... 1.9, 2
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
X_poly_2 = poly_reg.fit_transform(X_grid)

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.plot(X_grid, lin_reg_2.predict(X_poly_2), color='green')
plt.title("Truth or Bluff (Linear Reggresion) VS Polynomminal Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")

plt.show()

# predict salary for single pweson and his previous salary  with both models
print(lin_reg.predict([[6.5]]))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))



