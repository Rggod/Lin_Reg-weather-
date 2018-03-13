# linear regression model

# is there a relationship between water salinity and water temperature based on the data



# import the required libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# obtain the dataset
dataset = pd.read_csv('bottle.csv')

# split the dataset into the neccessary dependent and independent variable
X = dataset.iloc[:,6].values
Y = dataset.iloc[:,5].values
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)

# fill out the missing data(salinity and water temperature) by taking the mean of the other data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values ='NaN', strategy ='mean', axis =0)
imputer = imputer.fit(X[:,0:1])
X[:,0:1] = imputer.transform(X[:,0:1])
imputer = imputer.fit(Y[:,0:1])
Y[:,0:1] = imputer.transform(Y[:,0:1])


# split the data into test and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size =0.8)

# use the linear regression to correlation between the 2 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# predict using the regressor
Y_pred = regressor.predict(X_test)
Y_pred1 = regressor.predict(X_train)


# give a visual represenation of the data
plt.hexbin(X_train,Y_train, gridsize=25)
plt.plot(X_train,Y_pred1, color = 'green')
plt.title('temperature vs salinity')
plt.xlabel('salinity')
plt.ylabel('temperature(C)')
plt.show()

#############################################################