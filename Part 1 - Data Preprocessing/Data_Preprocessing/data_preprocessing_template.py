# Data Preprocessing Template

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
data_set = pd.read_csv("Data.csv")

# get features and labels
X = data_set.iloc[:,:-1].values
y = data_set.iloc[:,3].values

# handle missing values
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

imputer = imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_X = LabelEncoder()

X[:,0] = label_encoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features=[0])

X = onehotencoder.fit_transform(X).toarray()