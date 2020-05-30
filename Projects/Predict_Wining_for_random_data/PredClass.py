from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
import numpy as np
from sklearn.svm import SVR


class Pred:
    def __init__(self, regression_algorithim, n_train=800):
        """

        :List dataset: a 2d array
        first dimention is the date
        second dimention is the frequancy or the winning pre this date
        """
        self.n_train = n_train
        self.regressor = regression_algorithim

    def test(self, X, y):
        self.eval_on_features(X, y, self.regressor)

    # function to evaluate and plot a regressor on a given feature set
    def eval_on_features(self, features, target, regressor):
        # split the given features into a training and a test set
        X_train, X_test = features[:self.n_train], features[self.n_train:]
        # also split the target array
        y_train, y_test = target[:self.n_train], target[self.n_train:]
        regressor.fit(X_train, y_train)
        print("Train-set R^2: {:.2f}".format(regressor.score(X_train, y_train)))
        print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))
        y_pred = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)
        plt.figure(figsize=(10, 3))

        # plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90,
        #            ha="left")

        plt.plot(range(self.n_train), y_train, label="train")
        plt.plot(range(self.n_train, len(y_test) + self.n_train), y_test, '-', label="test")
        plt.plot(range(self.n_train), y_pred_train, '--', label="prediction train")

        plt.plot(range(self.n_train, len(y_test) + self.n_train), y_pred, '--',
                 label="prediction test")
        plt.legend(loc=(1.01, 0))
        plt.xlabel("Date")
        plt.ylabel("Rentals")


if __name__ == "__main__":
    df = pd.read_csv('dataset/alafdal_win_per_3_hours.csv')
    df = df.set_index(df['date'])
    df.index = pd.to_datetime(df.index)
    # df = df.iloc[400:800]
    citibike = df
    y = citibike['win']

    # regressor = LinearRegression()
    X_hour_week = np.hstack([
        citibike.index.month.values.reshape(-1, 1),
        citibike.index.days_in_month.values.reshape(-1, 1),
        citibike.index.week.values.reshape(-1, 1),
        # citibike.index.dayofyear.values.reshape(-1, 1),
        citibike.index.dayofweek.values.reshape(-1, 1),
        citibike.index.hour.values.reshape(-1, 1),
    ])
    poly_transformer = PolynomialFeatures(degree=3, interaction_only=True,
                                          include_bias=False)
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    # regressor = SVR(n_estimators=100, random_state=0)

    enc = OneHotEncoder()
    X_hour_week = enc.fit_transform(X_hour_week)
    # X_hour_week = poly_transformer.fit_transform(X_hour_week)

    pred = Pred(regressor, n_train=800)
    pred.test(X_hour_week, y)
    # eval_on_features(X_hour_week, y, regressor)
    # plt.plot(range(len(df.index)), df['count'])
