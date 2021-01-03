import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


drivers = pd.read_csv("./datasets/drivers.csv")
results = pd.read_csv("./datasets/results.csv")
races = pd.read_csv("./datasets/races.csv")
pit_stops = pd.read_csv("./datasets/pit_stops.csv")

concat_driver_name = lambda x: f"{x.forename} {x.surname}"

drivers['driver'] = drivers.apply(concat_driver_name, axis=1)

driver_names = drivers[['driverId', 'driver']]

results = results[['resultId', 'raceId', 'driverId', 'constructorId', 'grid', 'position', 'points']]
results = results[['resultId', 'raceId', 'driverId', 'constructorId', 'grid', 'position', 'points']]
results=results.replace('\\N',0)
merge1 = driver_names.merge(results, on='driverId')

races = races[races.year>2017]

races = races[['raceId', 'year', 'name', 'circuitId']]

merge2 = merge1.merge(races, on='raceId')

pit_stops = pit_stops[['raceId', 'driverId', 'stop']]

merge3 = merge2.merge(pit_stops, on=["raceId", "driverId"])

merge3 = merge3[['driverId', 'resultId', 'constructorId', 'raceId', 'grid', 'position', 'points', 'year', 'circuitId', 'stop']]

circuit_names = races[['circuitId', 'name']]

df_deneme = merge3

#shuffle
from sklearn.utils import shuffle
df_deneme = shuffle(df_deneme)


x = df_deneme.iloc[:, : -1]
y = df_deneme.iloc[:, -1]

#split dataset into train and test splits
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


#fit SVM model to training set
from sklearn.svm import SVC
svm = SVC(kernel = 'linear', random_state = 0) 
svm.fit(x_train, y_train)

#predict the results on test set
y_pred = svm.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

acc = sum(y_pred == y_test) / y_test.shape[0]

#k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = svm, X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std() #variance




"""
#cost versus accuracy
accuracy = np.zeros(10)
j = 0

for i in range(1, 50, 5):
   svm = SVC(kernel = 'linear', random_state = 0, C = i) 
   svm.fit(x_train, y_train) 
    
   y_pred = svm.predict(x_test)
    
   #finding error
   accuracy[j] = sum(y_pred == y_test) / y_test.shape[0]
   j = j + 1


plt.scatter(range(1, 50, 5), accuracy, color = 'red')
plt.plot(range(1, 50, 5), accuracy, color = 'blue')
plt.xlabel('cost')
plt.ylabel('accuracy')
plt.show()
"""

