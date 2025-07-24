import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("part_2_regresion/Section_8_Decision_Tree_Regression/Python/Position_Salaries.csv")
X = dataset.iloc[ : , 1 : -1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

from sklearn.tree import DecisionTreeRegressor
# we just set this random state parameter to have same results as in course clips
regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(X,y)

y_predict = regressor.predict([[6.5]])
print(y_predict)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid))
plt.title("Decision Tree Regresion Model Example")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()