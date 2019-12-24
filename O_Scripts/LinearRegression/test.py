import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from O_Scripts.ResidualPlot import ResidualPlot

X, y = datasets.load_boston(return_X_y=True)
clf = LinearRegression(n_jobs=-1)
clf.fit(X, y)
y_hat = clf.predict(X)

rp = ResidualPlot(y, y_hat)
rp.Y_hat_residual()
# rp.Y_residual()

# X = diabetes.data
# y = diabetes.target
#
# X = np.delete(X, 6, 1);
# X = np.delete(X, 2, 1);
# X2 = sm.add_constant(X)
# est = sm.OLS(y, X2)
# est2 = est.fit()
# print(est2.summary())
