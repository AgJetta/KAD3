import os
from itertools import combinations

import pandas as pd
import numpy as np
os.environ['OMP_NUM_THREADS'] = '1'
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

file_path = 'data.csv'

column_names = ["Długość działki kielicha (cm)", "Szerokość działki kielicha (cm)", "Długość płatka (cm)",
                "Szerokość płatka (cm)", "Gatunek"]

# Załadowanie datasetu
data = pd.read_csv(file_path, names=column_names)
# print(data)

# lista headerow
header_list = data.columns.tolist()

measurable_attributes = data.iloc[:, :4]

print(measurable_attributes)


# Perform k-means clustering using all four attributes
k = 3  # Assuming you want to cluster into 3 groups
kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(measurable_attributes)
labels = kmeans.labels_

# Preparing the combinations of the four features for plotting
feature_combinations = list(combinations(range(4), 2))

# Set up the matplotlib figure (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(12, 18))

# Flatten the array of axes for easy iterating
axes = axes.ravel()

for i, (col1, col2) in enumerate(feature_combinations):
    axes[i].scatter(measurable_attributes.iloc[:, col1], measurable_attributes.iloc[:, col2], c=labels)
    axes[i].set_xlabel(column_names[col1])
    axes[i].set_ylabel(column_names[col2])

plt.tight_layout()
plt.show()