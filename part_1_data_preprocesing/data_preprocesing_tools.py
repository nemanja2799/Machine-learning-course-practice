# libraries for tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer



# importing the dataset
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

# Handling missing data: There are several ways: 
# 1. If there is lot of data and we are missing 1% of data we can delete those data and that will not affect model
# 2. Second approach is to calculate average of all known data and use it for missing data 

# create object from class and then give him rule for missing values and strategy for replacing
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer.fit(x[:, 1:3]) # these calculates for data range we set here
x[:, 1:3] = imputer.transform(x[:, 1:3]) # these transform data for range we set here

print(x)
print(y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# OneHotEncoding
# transform values from textual categories(for example 3 languages: Spanish, German,Italian) in three categories with equal priority
# in values(1,2,4(001, 010, 100) to make it easier for machine learning model to process them)

# kind of transform way of transform and Clas of way of transformation,and thrid argument in transform is 
# index od column we want to transform, reminder-way we want to treat other column(we keep them)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# model expect array in input

x = np.array(ct.fit_transform(x))
print(x)

# for columns which containes values 'yes' or 'no' for some action like for example purchased we
# encode just 1 for yes and 0 for no
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# split models in train model in which we train model and test model in which we test model( test model is unknown for machine learning model until moment of test)
# good practise is to use 80 % for training and 20% for test model(test_size parameter)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
