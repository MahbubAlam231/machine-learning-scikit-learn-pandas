# ----------------------------------------------------------------
#!/usr/bin/python3
# Author(s)   : Mahbub Alam
# File        : KNN_fruit_detection.py
# Created     : 2025-04-04 (Apr, Fri) 13:39:36 CEST
# Description : X
# ----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # importing classifier constructor

fruits = pd.read_table('fruit_data_with_colors.txt')

print(fruits.head())

# print(fruits.columns) # Output: ['fruit_label', 'fruit_name', 'fruit_subtype', 'mass', 'width', 'height', 'color_score']

print(f"")

# ==================[[ check missing data ]]==================={{{

missing_data = fruits.isna().any()
cols_with_nan = fruits.columns[missing_data].to_list()
print(cols_with_nan)
print(f"")

# }}}

# ============[[ fruit label and name dictionary ]]============{{{

lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
# print(lookup_fruit_name) # Output: {1: 'apple', 2: 'mandarin', 3: 'orange', 4: 'lemon'}

# }}}

# ===================[[ train test split ]]===================={{{

X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# }}}

# =============[[ learning with KNN classifier ]]=============={{{

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)

print(f"Accuracy score: {knn.score(X_test, y_test)}")


example_fruits = [[20, 4.3, 5.1], [180, 7.8, 8.3]]
fruit_predictions = knn.predict(example_fruits)
print([lookup_fruit_name[label] for label in fruit_predictions])

# }}}

# =========[[ decision boudaries for KNN classifier ]]========={{{

from adspy_shared_utilities import plot_fruit_knn

# plot_fruit_knn(X_train, y_train, 5, 'uniform')

# }}}

# ===============[[ classifier accuracy vs k ]]================{{{

k_range = range(1, 20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.scatter(k_range, scores)
# Axis labels and title
plt.xlabel("k")
plt.ylabel("accuracy")
plt.xticks(range(0, 20, 5))
if 0:
    plt.show()
else:
    plt.close()

# }}}

# ========[[ classifier accuracy vs train/test split ]]========{{{

t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 5)

plt.figure()

for s in t:

    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy');
plt.show()

# }}}

