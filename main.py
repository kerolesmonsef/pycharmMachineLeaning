from O_Scripts.LinearRegression.OwnLinearRegression import OwnLinearRegression
from O_Scripts.LinearRegression.Gradient_Descent import Gradient_Descent
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model, preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

##########################
# Own Linear Regression #
########################

# own = OwnLinearRegression(seed=True)
# m, b = own.best_fit_slope_and_intercept()
# own.draw();


####################################################
# Own Gradient_Descent in Linear Regression #
################################################

XX, y = datasets.load_diabetes(return_X_y=True)
# X = XX[:, 2:5]
X = XX
X = preprocessing.scale(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
print("Built In", clf.intercept_, clf.coef_)
# print("---------------------")
# g_d = Gradient_Descent(x_train, y_train, l_r=0.0001, N_iter=1000000);
# g_d.fit();
# print("Gradient_Descent", g_d.theta)
# print('--------- Errors------------')
# print("Gradient R_Square", g_d.r_squar(x_test, y_test))
# clf_SSR = sum((clf.predict(x_test) - y_test) ** 2)
# clf_SST =  sum((y_test - np.average(y_test)) ** 2)
# print("CLF R_Square", 1-(clf_SSR/clf_SST))
