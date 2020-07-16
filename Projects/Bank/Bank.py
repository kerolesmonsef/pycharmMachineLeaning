import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

data = pd.read_csv('Churn-Modelling.csv')
y = data['Exited']
data.drop(labels=(['Exited', 'RowNumber', 'CustomerId', 'Surname']), axis=1, inplace=True)
X = data.copy()

X_Gender_Geography_hot = pd.get_dummies(X[['Geography', 'Gender']])

# le = LabelEncoder()
# X[['Geography', 'Gender']] = X[['Geography', 'Gender']].apply(le.fit_transform)
X[X_Gender_Geography_hot.columns] = X_Gender_Geography_hot[X_Gender_Geography_hot.columns]
X.drop(labels=(['Geography', 'Gender']), axis=1, inplace=True)

X_upsampled, y_upsampled = resample(X[y == 1],
                                    y[y == 1],
                                    replace=True,
                                    n_samples=X[y == 0].shape[0],
                                    random_state=123)

X = np.vstack((X[y == 0], X_upsampled))
y = np.hstack((y[y == 0], y_upsampled))

# X = PolynomialFeatures(degree=3).fit_transform(X)
# X = PCA(n_components=100).fit_transform(X)
# print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

print('begin')
logistic_regression = LogisticRegression(random_state=1, n_jobs=-1)

decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)

k_neighbors = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski', n_jobs=-1)

logistic_regression_pipe = Pipeline([['sc', StandardScaler()],
                                     ['clf', logistic_regression]])
KNeighbors_pipe = Pipeline([['sc', StandardScaler()],
                            ['clf', k_neighbors]])

ada_boost_classifier_pipe = Pipeline([
    ['clf', AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=1),
        n_estimators=500,
        learning_rate=0.1,
        random_state=1)]
])

svm_pipe = Pipeline([('sc', StandardScaler()),
                     ('clf', SVC())])

random_forest = Pipeline([
    ['clf', RandomForestClassifier(n_estimators=100, n_jobs=-1)]])

py_voting_classifier = VotingClassifier(estimators=[
    ('lr', logistic_regression_pipe), ('dt', decision_tree), ('KNN', KNeighbors_pipe), ('rf', random_forest),
    ('svm', svm_pipe)],
    voting='hard', n_jobs=-1)

clf_labels = ['Logistic regression', 'Decision tree', 'KNN', 'svm', 'AdaBoostClassifier',
              'Random Forest', 'Majority voting', 'py_voting_classifier']

all_clf = [logistic_regression_pipe, decision_tree, KNeighbors_pipe, svm_pipe, ada_boost_classifier_pipe,
           random_forest, py_voting_classifier]

for clf, label in zip(all_clf, clf_labels):
    scores = clf.fit(X_train, y_train).score(X_test, y_test)
    print(label, " : ", scores)

input('press any key to continue ....')
