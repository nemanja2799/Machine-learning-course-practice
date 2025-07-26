import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("part_2_regresion/Section_9_Random_Forest_Regression/Python/Position_Salaries.csv")
X = dataset.iloc[ : , 1 : -1].values
y = dataset.iloc[ : , -1].values

# ensamble is multiple usage of one same algoritham or more alghorithams
from sklearn.ensemble import RandomForestRegressor
# n_estimators is number of trees we want to create in model
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X,y)

print(regressor.predict([[6.5]]))


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Random forest regression")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()


