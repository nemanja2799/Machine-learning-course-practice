import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('part_2_regresion/Section_5_Multiple_Linear_Regression/Python/50_Startups.csv')

X = dataset.iloc[ : , : -1].values
y = dataset.iloc[ :, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# transformed x model will have 3 extra columns on begining of array

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

# in multiple linear regresion there is no need for feature scaling because every x variable
# has its own coeficient, so it compensate if there is much higher values
# in some columns than others

# we dont check linear assumption on multiple linear model it is waste of time

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Calculate profits in y_test with our model
y_pred = regressor.predict(X_test)

# Set to peint value with two decimal places
np.set_printoptions(precision=2)

# to have more beautiful print we will concatenate real profit and predicted profit for y_test(y_pred)
# reshape make vertical vector from horisontal(first element is number of rows we want in result,and second is number of columns)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
# we use horisontal concatenation here because we want to have these columns together in same row

# predict for single input with our model
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# get and print coefficients for model

print(regressor.coef_)
print(regressor.intercept_)