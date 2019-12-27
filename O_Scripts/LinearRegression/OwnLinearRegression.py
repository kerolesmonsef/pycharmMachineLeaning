from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random


class OwnLinearRegression:
    xs = np.array([], dtype=np.float64)
    ys = np.array([], dtype=np.float64)
    m = 0
    b = 0

    def __init__(self, seed=False):
        if seed is True:
            random.seed(10)
        self.xs, self.ys = self.S_Linear_R_Random_Dataset(300, 50, 1, Positive=False)

    def S_Linear_R_Random_Dataset(self, N, variance, step=2, Positive=True):
        val = 1
        ys = []
        for _ in range(N):
            y = val + random.randrange(-variance, variance)
            ys.append(y)
            if Positive is True:
                val += step
            elif Positive is False:
                val -= step
        xs = [i for i in range(N)]
        return np.array(xs, np.float64), np.array(ys, np.float64)

    def best_fit_slope_and_intercept(self):
        xs = self.xs
        ys = self.ys
        M = ((mean(xs) * mean(ys)) - mean(xs * ys)) / (mean(xs) ** 2 - mean(xs ** 2))
        b = mean(ys) - M * mean(xs)
        self.m = M
        self.b = b
        return M, b

    def cof_of_deter(self, y_orig, y_line):
        y_mean_line = [mean(y_line) for _ in y_line]
        squared_error_regr = self.squared_error(y_orig, y_line)
        squared_error_y_mean = self.squared_error(y_orig, y_mean_line)
        return 1 - (squared_error_regr / squared_error_y_mean)

    def squared_error(self, y_orig, y_line):
        return sum((y_orig - y_line) ** 2)

    def draw(self):
        regression_line = [(self.m * x) + self.b for x in self.xs]
        r_squared = self.cof_of_deter(self.ys, regression_line)
        print("Slope = ", self.m, "b=", self.b, "r_squared", r_squared)

        plt.scatter(self.xs, self.ys, color='red')
        plt.plot(self.xs, regression_line, color='black')
        plt.show()
