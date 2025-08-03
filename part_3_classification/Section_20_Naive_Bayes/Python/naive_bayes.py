import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


dataset = pd.read_csv("part_3_classification/Section_19_Kernel_SVM/Python/Social_Network_Ads.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc =StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#svc is suport vector classification
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
