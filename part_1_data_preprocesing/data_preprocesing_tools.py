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
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# feature scaling not necessary for all models - explained later
# 2 types: standardisation and normalisation 
# standardisation x-mean(x)^&*))/(standard deviatioon(x)) - all values betveen +-3 / used for all models
# normalisation x-min(x/(max(x) - min(x))) all values between 0 and 1 / used for specific models

# this should be applied to models after splitting to test and train models, so there will be two scaling for each model
# standardization shouldn't be applied on columns which is transformed with OneHotEncoder or LinearEncoder,
# because we will lose information we got with theese methods
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
# first we make scaler with train model and then just apply this scaler on test model
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)