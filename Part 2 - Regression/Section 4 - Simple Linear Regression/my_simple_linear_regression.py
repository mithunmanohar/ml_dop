# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 01:10:44 2018

@author: mmanoh2x
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data

data  = pd.read_csv("Salary_Data.csv")

X = data.iloc[:, :-1].values
y =  data.iloc[:,1].values
# Split data into training and test data

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, 
                                                    random_state=0)


# Create model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train) # the linear regressor learns the correlation
                                # of the training sets by fitting on the data

y_pred = regressor.predict(X_test) # a vector of predictions of dependant variables

# y_pred is the predicted salary and y_test is the actual salary                              

# Plot the data

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue') # to comapre between actual salary andpredicted salary for train set
plt.xlabel("years_of_experience")
plt.ylabel("salary")
plt.show()


plt.scatter(X_test, y_test)
plt.xlabel("years_of_experience")
plt.ylabel("salary")
plt.show()
