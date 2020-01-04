import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


filename = 'Data/Classification/naive_bayes_pandas.data'
df = pd.read_csv(filename)
X = df.drop(df.columns[-1], 1)
y = df[df.columns[-1]]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=21)

clf = GaussianNB()
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
pred = clf.predict([[4.6, 3.1, 1.5, 0.2]])

print("score : ", score)
