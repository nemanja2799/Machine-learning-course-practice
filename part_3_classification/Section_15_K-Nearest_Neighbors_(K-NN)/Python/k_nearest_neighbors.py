import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("part_3_classification/Section_15_K-Nearest_Neighbors_(K-NN)/Python/Social_Network_Ads.csv")
X = dataset.iloc[ : , : -1].values
y = dataset.iloc[ : , -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
# as parameter we set number of K neibgours we want to use in opservations
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train,y_train)
print(classifier.predict(sc.transform([[30, 87000]])))
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape((len(y_pred),1)), y_test.reshape((len(y_test),1))), 1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
