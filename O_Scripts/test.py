import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.utils import resample

data = pd.read_csv('../Projects/Bank/Churn-Modelling.csv')
y = data['Exited']
data.drop(labels=(['Exited', 'RowNumber', 'CustomerId', 'Surname']), axis=1, inplace=True)
X = data.copy()
X_Gender_Geography_hot = pd.get_dummies(X[['Geography', 'Gender']])

X[X_Gender_Geography_hot.columns] = X_Gender_Geography_hot[X_Gender_Geography_hot.columns]
X.drop(labels=(['Geography', 'Gender']), axis=1, inplace=True)

X_upsampled, y_upsampled = resample(X[y == 1],
                                    y[y == 1],
                                    replace=True,
                                    n_samples=X[y == 0].shape[0],
                                    random_state=123)

X = np.vstack((X[y == 0], X_upsampled))
y = np.hstack((y[y == 0], y_upsampled))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=None,
                              random_state=1)

bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs=-1,
                        random_state=1)

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f'
      % (tree_train, tree_test))

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)

bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f'
      % (bag_train, bag_test))
