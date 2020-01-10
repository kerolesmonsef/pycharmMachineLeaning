import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df = pd.read_csv('Data/Classification/breast_cancer_wisconsin.csv')
df.replace('?', -99999, inplace=True)
X = df.drop(['class'], 1)
y = df['class']
# lda = LinearDiscriminantAnalysis()
# X = lda.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("accuracy", accuracy)
# example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 1, 1, 2, 3, 2, 1]])
# prediction = clf.predict(example_measures)
# print(prediction)
