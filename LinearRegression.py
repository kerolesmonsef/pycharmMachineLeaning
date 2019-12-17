import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Data/wiki.csv")
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forcast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forcast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forcast_col].shift(-forcast_out)  # the DF contain Nan In the End
X = np.array(df.drop('label', 1))
X = preprocessing.scale(X)
X = X[:-forcast_out]  # The X does Not Contain NaN
X_lately = X[-forcast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])
# X = preprocessing.scale(X)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(accuracy)
forcast_set = clf.predict(X_lately)
print(forcast_set)