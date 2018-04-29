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

# building optimal model

"""
Backward Elimination : 
    - Sometimes not all independant variables are not as statistically 
    significant as others. Some has great impact/ effect on the dependant 
    variables, butothers might have less impact.
    
    - So even if we remove the non-statistically significant variables, we may 
    get very good predictions.
    
    - The goal here is to find an optimal team of independant variables, so
    that each independant variable of the team has a great impact on the 
    dependant variable.
    
    - ie each independant variable is a strong predictor of dependant variable
    and is statistically significant.
    
    - This can be positive - increase in 1 unit of independant variable : the 
    dependant variable increase or it can be negative.
    
    
"""

# backward elimination preperation

import statsmodels.formula.api as sm

"""

- The equation for multiple linear regression is 
    
    y = b0 + b1X1 + b2X2 + ... + bnXn
    
    - b0 is a constant which is not assosiated with any independant variable.
    - But we can assosiate a constant X0 = 1 to the b0 in which case
    
    y = b0X0 + b1X1 + b2X2 + ... + bnXn
    
    The stats model we are going to use do not account for this b0 constant.
    
- So we are going to add this column of ones it to our matrics of 
  independant variable


- Below appends a 50*1 array to the matrics of type int in the axis = 1(row)

"""

# X = np.append(X, values = np.ones((50, 1).astype(int), axis=1)

"""
This adds one to the last. To add it to first do the below - reverse the 
ones matrics and original value X

"""

X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis=1)

# backward elimination

import statsmodels.formula.api as sm

X_opt = X[:, [0, 1, 2 , 3, 4, 5]]


regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

"""
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
x3             0.8060      0.046     17.369      0.000       0.712       0.900
x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
x5             0.0270      0.017      1.574      0.123      -0.008       0.062
"""

# Eliminate X2


X_opt = X[:, [0, 1 , 3, 4, 5]]


regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

"""
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.011e+04   6647.870      7.537      0.000    3.67e+04    6.35e+04
x1           220.1585   2900.536      0.076      0.940   -5621.821    6062.138
x2             0.8060      0.046     17.606      0.000       0.714       0.898
x3            -0.0270      0.052     -0.523      0.604      -0.131       0.077
x4             0.0270      0.017      1.592      0.118      -0.007       0.061
"""

# Eliminate X1

X_opt = X[:, [0 , 3, 4, 5]]


regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

"""
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.012e+04   6572.353      7.626      0.000    3.69e+04    6.34e+04
x1             0.8057      0.045     17.846      0.000       0.715       0.897
x2            -0.0268      0.051     -0.526      0.602      -0.130       0.076
x3             0.0272      0.016      1.655      0.105      -0.006       0.060
==============================================================================
Omnibus:                       14.838   Durbin-Watson:                   1.282
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.442
Skew:                          -0.949   Prob(JB):                     2.21e-05
Kurtosis:                       5.586   Cond. No.                     1.40e+06
==============================================================================

"""

# Eliminate X2

X_opt = X[:, [0 , 3, 5]]


regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

"""
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
x1             0.7966      0.041     19.266      0.000       0.713       0.880
x2             0.0299      0.016      1.927      0.060      -0.001       0.061
==============================================================================
Omnibus:                       14.677   Durbin-Watson:                   1.257
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.161
Skew:                          -0.939   Prob(JB):                     2.54e-05
Kurtosis:                       5.575   Cond. No.                     5.32e+05
==============================================================================

"""


"""
Lower the pValue : More significant is your independant variable with respect
to the dependant Variable

"""

X_opt = X[:, [0, 3]]


regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()
"""
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.903e+04   2537.897     19.320      0.000    4.39e+04    5.41e+04
x1             0.8543      0.029     29.151      0.000       0.795       0.913
==============================================================================
"""

