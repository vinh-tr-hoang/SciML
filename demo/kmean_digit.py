"""
Clustering for classification of handwritten digits using scikit-learn

"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
digits = np.array (pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None))

plt.figure ()
plt.imshow(digits[0,1:].reshape((8,8)), cmap ='Grey')

# Code Start here:
num_clusters = list(range (7,11) )
inertias = []
for i in range (len(num_clusters)):
  model = KMeans(num_clusters[i])
  model.fit(digits)
  inertias.append(model.inertia_/digits.shape[1])
  print (inertias)
plt.figure()
plt.plot(num_clusters, inertias, '-o')

for i in range(num_clusters[i]):
  plt.figure (10 + i )
  plt.imshow(model.cluster_centers_[i,1:].reshape((8,8)), cmap ='Grey')
plt.show()
