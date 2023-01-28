# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 19:07:03 2023

@author: antho
"""

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('nba_stats&salaries&all_star_status.csv')


X = df.drop(['Salary','Player'], axis=1)
Y = df.Salary



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train.shape, Y_train.shape


X_test.shape, Y_test.shape

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

model = linear_model.LinearRegression()

model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))

pickle.dump(model, open('model.pkl','wb'))






