import matplotlib.pyplot as plt
from sklearn import datasets, tree
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame({'Feature 1': X[:, 0],
                   'Feature 2': X[:, 1],
                   'Label': y})

# check if any data is missing.
missing_data = df.isnull().sum()

# only use first 2 features
X = X[:100, [0, 1]]
y = y[:100]

# string labels 
string_y = np.where(y == 0, 'Iris-Setosa', 'Iris-Versicolor')

dt = tree.DecisionTreeClassifier()
dt = dt.fit(X, y)

# checking predictions given a 2D array
print(dt.predict([[5.1, 3.5]]))
print(dt.predict([[7.0, 3.2]]))

# get preds for accuracy_score
y_pred = dt.predict(X)
acc = accuracy_score(y, y_pred)
print(f'Acc Score: {acc:.2f}%')

# visualize tree
fig = plt.figure(figsize=(20, 10))
dt_tree = tree.plot_tree(dt)
plt.show()

