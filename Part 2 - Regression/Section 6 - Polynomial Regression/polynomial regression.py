# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:11:33 2018

@author: mmanoh2x
"""

# Polynomial Linear Regression
"""
Regressions:
    
    - SIMPLE LINEAR REGRESSION     : y = b0 + b1X1
    
    - MULTIPLE LINEAR REGRESSION   : y = b0 + b1X1 + b2X2 + ... + bnXn
    
    - POLYNOMIAL LINEAR REGRESSION : y = b0 + b1X1 + b2X1^2 + .. + bnX1^n
    
- Polynomial regression is used to fit the data with curves. Its still called 

Linear regression : becuase even thought the relation between the dependant and
independant variables is non-linear, we are trying to express the dependant 
variables as 
 
   - the linear combination of coifficients and independant variables
        
       y = b0 + b1X1 + b2X1^2 + .. + bnX1^n

"""

# Polynomial regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
import os
print (os.getcwd())
data_set = pd.read_csv("Position_Salaries.csv")

print (data_set.head())

X = data_set.iloc[:, 1:2].values
y = data_set.iloc[:, 2:3].values

""" 
# split data to train and test

Splitting data to train and test : is not required here as the dataset
is very small
"""

"""
# Feature scaling

from sklearn.preprocessing import StandardScalar
sc_X = StandardScalar()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

Feature scaling : is not required as we are using the same library used 
for linear regression, which automatically does the feature scaling for more 
accurate predictions

"""

# Fit linear Regression model to dataset

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)



#













