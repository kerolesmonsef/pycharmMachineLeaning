import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Load data
def sigmoid_T(x):
    return 1 / (1 + np.exp(-x));


def sigmoid_F(x):
    return 1 / (1 + np.exp(x));


data = np.loadtxt('../../Data/Classification/heights_weights.csv', delimiter=',', skiprows=1)
X = data[:, 1:]
y = data[:, 0]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Fit (train) the Logistic Regression classifier
clf = LogisticRegression(C=1e40, solver='newton-cg', n_jobs=-1)
fitted_model = clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print("Acuracy : ", score)
# Predict
prediction_result = clf.predict([(70, 180), (71, 180)])
print("prediction_result", prediction_result)
x = np.array([1, 70, 180])
x1 = np.array([1, 71, 180])
theta = np.array([-0.17578539, -0.47455781, 0.19622027])
Function = np.sum(x * theta)
prob_T = sigmoid_T(Function)
prob_F = sigmoid_F(Function)
print(prob_T)
