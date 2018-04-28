# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:22:10 2018

@author: mmanoh2x
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load dataset 

data_set = pd.read_csv("50_Startups.csv")

X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values

# encode the data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()

X[:,3] = label_encoder.fit_transform(X[:,3])


one_hot_encoder = OneHotEncoder(categorical_features = [3])

X = one_hot_encoder.fit_transform(X).toarray()

# avoid dummy trap variable

X = X[:, 1:]

# split data to train and test data

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)

# predicting test result

y_pred = regressor.predict(X_test)



