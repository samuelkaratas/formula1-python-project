# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 03:46:28 2021

@author: Erkin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

drivers = pd.read_csv("./datasets/drivers.csv")
results = pd.read_csv("./datasets/results.csv")
races = pd.read_csv("./datasets/races.csv")
pit_stops = pd.read_csv("./datasets/pit_stops.csv")

concat_driver_name = lambda x: f"{x.forename} {x.surname}"

drivers['driver'] = drivers.apply(concat_driver_name, axis=1)
driver_names = drivers[['driverId', 'driver']]
results = results[['resultId', 'raceId', 'driverId', 'constructorId', 'grid', 'position', 'points']]
results=results.replace('\\N',0)
results['position'] = results['position'].astype(int)

merge1 = driver_names.merge(results, on='driverId')
races = races[races.year>2017]
races = races[['raceId', 'year', 'name', 'circuitId']]
merge2 = merge1.merge(races, on='raceId')
pit_stops = pit_stops[['raceId', 'driverId', 'stop']]
merge3 = merge2.merge(pit_stops, on=["raceId", "driverId"])
merge3 = merge3[['driverId', 'resultId', 'constructorId', 'raceId', 'grid', 'position', 'points', 'year', 'circuitId', 'stop']]
circuit_names = races[['circuitId', 'name']]

df_project = merge3

x = df_project.iloc[:, [4,9]]
y = df_project.iloc[:, 5]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.tree import DecisionTreeClassifier 
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print(confusion_mat)



from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha = 0.75, cmap = ListedColormap(('blue', 'yellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('black', 'white','white','white','white','white','red'))(i), label = j)
plt.title('Decision Tree (Test set)')
plt.xlabel('Position gained')
plt.ylabel('Number of pitstops')
plt.legend()
plt.show()
 
 
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'yellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
       plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
               c = ListedColormap(('black', 'white','white','white','white','white','red'))(i), label = j)
plt.title('Decision Tree(Training Set))')
plt.xlabel('Position gained')
plt.ylabel('Number of pitstops')
plt.legend()
plt.show()