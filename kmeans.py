import os
from itertools import combinations
import colormaps as cmaps
import mcmd
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score



os.environ['OMP_NUM_THREADS'] = '1'
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as mcm  # Importing the cm module
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

species_column = data['Gatunek']

print(measurable_attributes)


def check_with_sklearn():
    # Preparing the combinations of the four features for plotting
    feature_combinations = list(combinations(range(4), 2))

    # Range of k values to try
    k_values = range(1, 11)


    for k in k_values:
        # Perform k-means clustering using all four attributes
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(measurable_attributes)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_  # Cluster centers

        # Set up the matplotlib figure (3 rows, 2 columns)
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        plt.suptitle(f'K-Means Clustering with k={k} \n')

        # Flatten the array of axes for easy iterating
        axes = axes.ravel()

        # Use a colormap for better color distinction
        cmap = mcm.nipy_spectral  # Directly using the colormap

        for i, (col1, col2) in enumerate(feature_combinations):
            scatter = axes[i].scatter(measurable_attributes.iloc[:, col1], measurable_attributes.iloc[:, col2], c=labels,
                                      cmap=cmap, alpha = 0.8)
            axes[i].set_xlabel(column_names[col1])
            axes[i].set_ylabel(column_names[col2])
            # axes[i].scatter(centers[:, col1], centers[:, col2], s=50, alpha=0.9, marker='D', color = 'black')


            # Add a color bar
        fig.subplots_adjust(bottom=0.1, top=0.9)  # Adjust the layout to make room for the colorbar
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # Position of the colorbar
        plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal')


        plt.show()

# check_with_sklearn()



# Function to calculate the Euclidean distance between points
def euclidean_distance(a, b):
    sum_squared_diff = sum((x - y) ** 2 for x, y in zip(a, b))
    return sum_squared_diff ** 0.5


# Initialize centroids by randomly selecting 'k' samples from the dataset
def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices, :]


# Assign data points to the nearest centroid
def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for idx, point in enumerate(data):
        closest_centroid = np.argmin([euclidean_distance(point, centroid) for centroid in centroids])
        clusters[closest_centroid].append(idx)
    return clusters


# Update centroid positions based on the mean of the assigned points
def calculate_new_centroids(data, clusters):
    return np.array([np.mean(data[cluster], axis=0) for cluster in clusters])



# Main k-means clustering function
def k_means(data, k, max_iters=10, tolerance=0):
    centroids = initialize_centroids(data, k)
    prev_centroids = centroids.copy()
    it = 0  # Initialize iteration counter
    for it in range(1, max_iters + 1):
        clusters = assign_clusters(data, centroids)
        centroids = calculate_new_centroids(data, clusters)

        # Check for convergence (if centroids do not change)
        diff = centroids - prev_centroids
        if np.all(np.abs(diff) <= tolerance):
            break
        prev_centroids = centroids.copy()

    labels = np.concatenate([np.full(len(cluster), i) for i, cluster in enumerate(clusters)])

    wcss = 0
    for cluster_idx, cluster in enumerate(clusters):
        centroid = centroids[cluster_idx]
        for idx in cluster:
            wcss += np.sum((data[idx] - centroid) ** 2)

    # Additional values to return
    iterations_values = it
    wcss_values = wcss

    return clusters, centroids, labels, it, wcss



####

k = 3

data_np = measurable_attributes.to_numpy()
feature_names = measurable_attributes.columns  # Get the feature names

clusters, centroids, labels, it, wcss = k_means(data_np, k)

# Preparing the combinations of the first four features for plotting
feature_combinations = list(combinations(range(4), 2))

# Set up the matplotlib figure (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(12, 18))
plt.suptitle(f'K-Means Clustering with k={k}')

# Flatten the array of axes for easy iterating
axes = axes.ravel()

colors = mcm.tab20(np.linspace(0, 1, k))

for i, (col1, col2) in enumerate(feature_combinations):
    scatter = axes[i].scatter(data_np[:, col1], data_np[:, col2], c=labels, cmap=mcm.tab20)

    axes[i].set_xlabel(column_names[col1])
    axes[i].set_ylabel(column_names[col2])

    # Plot centroids on the same scatter plot
    for cluster_index, cluster in enumerate(clusters):
        points = data_np[cluster]
        color = mcm.tab20(cluster_index / (k - 1))  # Adjust color based on cluster_index
        axes[i].scatter(points[:, col1], points[:, col2], color=color, label=f'Cluster {cluster_index}', s=100)
        axes[i].scatter(centroids[cluster_index, col1], centroids[cluster_index, col2],
                        s=100, color=color, marker='D', edgecolors='black')  # Same color for centroids

plt.savefig('clustering')

plt.show()


def iterations_vs_k(data, k_max):
    k_values = list(range(2, k_max + 1))
    iterations_values = [k_means(data, k)[3] for k in k_values]  # Use index 3 for iterations

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, iterations_values, marker='o', linestyle='-', color='forestgreen')
    plt.title('Wykres zależności liczby iteracji od k')
    plt.xlabel('Wartość k (liczba klastrów)')
    plt.ylabel('Liczba iteracji')
    plt.grid(True)

    plt.savefig('wcss_vs_it')

    plt.show()


def calculate_wcss(data, k):
    _, _, _, _, wcss = k_means(data_np, k)
    return wcss

def plot_wcss_vs_k(data, k_max):
    k_values = list(range(2, k_max + 1))
    wcss_values = [calculate_wcss(data, k) for k in k_values]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss_values, marker='o', linestyle='-', color='dodgerblue')
    plt.title('Wykres zależności WCSS od k')
    plt.xlabel('Wartość k (liczba klastrów)')
    plt.ylabel('Wartość WCSS')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(list(range(2, k_max + 1)))  # Set x-axis ticks for each k value

    # Improve grid for y-axis
    plt.yticks(np.arange(10, max(wcss_values) + 10, 10))  # Set y-axis ticks at intervals of 100
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)  # Add grid lines for y-axis

    plt.savefig('wcss_vs_k')

    plt.show()


plot_wcss_vs_k(measurable_attributes.to_numpy(), 10)
iterations_vs_k(measurable_attributes.to_numpy(), 10)

ari = adjusted_rand_score(species_column, labels)
ami = adjusted_mutual_info_score(species_column, labels)
nmi = normalized_mutual_info_score(species_column, labels)

print(f'Data for species column and k-means clusters')
print(f'ARI: {ari}')
print(f"AMI: {ami}")
print(f"NMI: {nmi}")
print('\n')