from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression
X, y = make_regression(
    n_samples=200, n_features=2, noise=4.0, random_state=0)
reg = RANSACRegressor(random_state=0).fit(X, y)
reg.score(X, y)

reg.predict(X[:1,])