import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("part_3_classification/Section_14_Logistic_Regression/Python/Social_Network_Ads.csv")
X = dataset.iloc[ : , : -1].values
y = dataset.iloc[ : , -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# we sholud do feature scaling on all columns of x data set 
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])

print(X_train)
print(X_test)

# import model for Logistic regression
from sklearn.linear_model import LogisticRegression

# for now we don't change parameters of model we will tune model parameters in part 10
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
print(classifier.predict(sc.transform([[30, 87000]])))

# there is method predict_proba which predict probability of y to happen
# and predict says strictly if it happend(0 not, 1 yes)
y_pred = classifier.predict(X_test)
# last parameter say if we want horisontal - 1 or vertical - 0 concatenation 
# both arrays is one parameter so they goes to () !!!
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test) ,1)), 1))

# to show number of correct and wrong predictions between y_pred and y_test we make 
# confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

# we calculate accuracy of prediction in percentage %(*100 for percent) successful prediction
print(accuracy_score(y_test, y_pred))
