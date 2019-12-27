from sklearn import datasets, linear_model
import numpy as np


class Gradient_Descent:
    xs = np.array([])
    ys = np.array([])
    xs = np.array([])
    xs_T = np.array([])
    theta = np.array([])
    l_r = 0.01
    N_iter = 1000

    def __init__(self, xs, ys, l_r=0.0001, N_iter=10000):
        self.xs = np.insert(xs, 0, values=1, axis=1)  # add extra column for the b =1
        self.ys = np.array(ys)
        self.l_r = l_r
        self.N_iter = N_iter
        _, var_num = xs.shape
        self.theta = np.zeros(var_num + 1)
        self.xs_T = np.transpose(self.xs)

    def __Predicted_Ys(self):
        Y_predicted = np.dot(self.xs, self.theta)
        return Y_predicted

    def fit(self):
        past_costs = []
        N = len(self.xs)
        for i in range(self.N_iter):
            predicted_Ys = self.__Predicted_Ys();
            self.theta = self.theta - (-2 / N) * self.l_r * (np.dot(self.xs_T, (self.ys - predicted_Ys)))

    def predict(self, p_xs):
        one_p_xs = np.insert(p_xs, 0, values=1, axis=1)
        y_pred = np.dot(one_p_xs, self.theta)
        return y_pred

    def SSR(self, xs, ys):
        return sum((self.predict(xs) - ys) ** 2)

    def SST(self, ys):
        avg_y = np.average(ys)
        return sum((ys - avg_y) ** 2)

    def r_squar(self, xs, ys):
        return 1 - self.SSR(xs, ys) / self.SST(ys);
