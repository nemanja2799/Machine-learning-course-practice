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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# transform values from textual categories(for example 3 languages: Spanish, German,Italian)
# in values(1,2,4(001, 010, 100) to make it easier for machine learning model to process them)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
