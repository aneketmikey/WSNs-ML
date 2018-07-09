import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
np.set_printoptions(precision=2)
weatherdata = pd.read_csv('Book1.csv')
feature_names_weatherdata = ['Temperature (C)', 'Apparent Temperature (C)', 'H
umidity', 'Wind Speed (km/h)','Wind Bearing (degrees)','Visibility (km)','Pres
sure (millibars)']
X_weatherdata = weatherdata[feature_names_weatherdata]
y_weatherdata = weatherdata['label']
target_names_weatherdata = ['Partly Cloudy', 'Mostly Cloudy', 'Overcast', 'Fog
gy','Breezy and Mostly Cloudy']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_weatherdata, y_weath
erdata, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train1)
# we must apply the scaling to the test set that we computed for the training
set
X_test_scaled = scaler.transform(X_test1)
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train_scaled, y_train1)
print('Accuracy of K-NN classifier on training set: {:.2f}'
.format(knn.score(X_train_scaled, y_train1)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
.format(knn.score(X_test_scaled, y_test1)))
example_weatherdataset = [[8.336, 10.355, 0.9, 26.15,300,1.35,1002.35]]
example_weatherdataset_scaled = scaler.transform(example_weatherdataset)
print('Predicted weather type for ', example_weatherdataset, ' is ',
target_names_weatherdata[knn.predict(example_weatherdataset_scaled)[
0]-1])
X_R1 = pd.read_csv('Book7.csv')
y_R1= pd.read_csv('Book6.csv')
from sklearn.neighbors import KNeighborsRegressor
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state =
0)
knnreg = KNeighborsRegressor(n_neighbors = 3).fit(X_train, y_train)
print(knnreg.predict(X_test))
print('R-squared test score: {:.3f}'
.format(knnreg.score(X_test, y_test)))
fig, subaxes = plt.subplots(5, 1, figsize=(8,30))
X_predict_input = np.linspace(0, 15, 500).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1,
random_state = 0)
for thisaxis, K in zip(subaxes, [1, 3, 5, 7, 15]):
knnreg = KNeighborsRegressor(n_neighbors = K).fit(X_train, y_train)
y_predict_output = knnreg.predict(X_predict_input)
train_score = knnreg.score(X_train, y_train)
test_score = knnreg.score( X_test, y_test)
thisaxis.plot(X_predict_input, y_predict_output)
thisaxis.plot(X_train, y_train, 'o', alpha=0.9, label='Train')
thisaxis.plot(X_test, y_test, '^', alpha=0.9, label='Test')
thisaxis.set_xlabel('Input feature')
thisaxis.set_ylabel('Target value')
thisaxis.set_title('KNN Regression (K={})\n\
Train $R^2 = {:.3f}$, Test $R^2 = {:.3f}$'
.format(K, train_score, test_score))
thisaxis.legend()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()