from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random


# plt.style.use('fivethirtyeight')


# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def S_Linear_R_Random_Dataset(N, variance, step=2, Positive=True):
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


xs, ys = S_Linear_R_Random_Dataset(300, 50, 1,Positive=False)


def best_fit_slope_and_intercept(xs, ys):
    M = ((mean(xs) * mean(ys)) - mean(xs * ys)) / (mean(xs) ** 2 - mean(xs ** 2))
    b = mean(ys) - M * mean(xs)
    return M, b


def cof_of_deter(y_orig, y_line):
    y_mean_line = [mean(y_line) for _ in y_line]
    squared_error_regr = squared_error(y_orig, y_line)
    squared_error_y_mean = squared_error(y_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


def squared_error(y_orig, y_line):
    return sum((y_orig - y_line) ** 2)


m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m * x) + b for x in xs]
r_squared = cof_of_deter(ys, regression_line)
print("Slope = ", m, "b=", b, "r_squared", r_squared)

plt.scatter(xs, ys, color='red')
plt.plot(xs, regression_line, color='black')
plt.show()
