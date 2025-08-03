import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from genieclust import compare_partitions
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Directories for each method
save_dirs = {
    'IsUMAP': './../results/MNIST_PSI/MNIST_isumap_csv',
    'Isomap': './../results/MNIST_PSI/MNIST_isomap_csv/',
    'UMAP': './../results/MNIST_PSI/MNIST_umap_csv/'
}

# MNIST
train = pd.read_csv('./../../Dataset_files/mnist_train.csv')
test = pd.read_csv('./../../Dataset_files/mnist_test.csv')
y_train = train['label']

X_train = train.drop('label', axis=1)
y_test = test['label']
X_test = test.drop('label', axis=1)

X_train['label'] = y_train
X = X_train.sample(n=10000, random_state=42)

y = X['label']
X = X.drop('label', axis=1)

original_labels = y.values

# Load reduced data and labels for each dimension
D = [400, 300, 200,150, 100, 50, 30, 10, 5, 2]

# Initialize a DataFrame to store PSI values for all methods
psi_data = pd.DataFrame()

# Part 1: PSI Calculation and Plotting for all Methods
for method, save_dir in save_dirs.items():
    psi_values = []
    
    for d in D:
        # Load reduced data and labels for the current method and dimension
        reduced_data = np.loadtxt(f'{save_dir}/reduced_{method.lower()}_d_{d}.csv', delimiter=',')
        reduced_labels = pd.read_csv(f'{save_dir}/reduced_{method.lower()}_labels_d_{d}.csv')['label'].values
        
        # Compute K-means clustering
        kmeans = KMeans(n_clusters=10, init='k-means++', random_state=42)
        kmeans.fit(reduced_data)
        kmeans_labels = kmeans.labels_

        # Compute PSI
        psi_value = compare_partitions.pair_sets_index(reduced_labels, kmeans_labels)
        psi_values.append(psi_value)
    
    # Add PSI values to the DataFrame
    method_data = pd.DataFrame({
        'Dimension': D,
        'PSI': psi_values,
        'Method': method
    })
    psi_data = pd.concat([psi_data, method_data], ignore_index=True)

# Print the resulting PSI DataFrame
print(psi_data)

# Plotting PSI values using seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Plot PSI values for each method on the same figure
sns.lineplot(x='Dimension', y='PSI', hue='Method', data=psi_data, marker='o')

# Customize plot appearance
plt.xlabel('Dimensionality')
plt.ylabel('Pair Sets Index (PSI)')
plt.title('Pair Sets Index (PSI) across Different Dimensionality Conditions for IsUMAP, Isomap, and UMAP')
plt.grid(True)

plt.xscale('log')
# Set the x-axis ticks explicitly
D2 = [400, 300, 200,150, 100, 50, 30, 10, 5, 2]
plt.xticks(D2, [str(d) for d in D2])


# Save the combined PSI plot
output_path_psi_combined = './../results/PSI/PSI_combined_plot.png'
plt.savefig(output_path_psi_combined, bbox_inches='tight', dpi=300)

# Show the combined PSI plot
plt.show()

# Part 2: 2D Visualization with K-means Centroids for All Methods
for method, save_dir in save_dirs.items():
    reduced_2D_data = np.loadtxt(f'{save_dir}/reduced_{method.lower()}_d_{2}.csv', delimiter=',')
    reduced_2D_labels = pd.read_csv(f'{save_dir}/reduced_{method.lower()}_labels_d_{2}.csv')['label'].values

    # K-means clustering for d=2 to get centroids
    kmeans_2d = KMeans(n_clusters=10, init='k-means++', random_state=42)
    kmeans_2d.fit(reduced_2D_data)
    centroids_2d = kmeans_2d.cluster_centers_

    print(f'{method} Confusion Matrix:', confusion_matrix(reduced_2D_labels, kmeans_2d.labels_))

    # Assign specific colors to each cluster
    cluster_colors = {
        0: 'green',
        1: 'blue',
        2: 'yellow',
        3: 'pink',
        4: 'purple',
        5: 'orange',
        6: 'brown',
        7: 'cyan',
        8: 'magenta',
        9: 'red'
    }

    # Plot the reduced data points with assigned colors
    fig, ax = plt.subplots(figsize=(12, 10))
    for cluster in range(10):
        indices = reduced_2D_labels == cluster
        ax.scatter(reduced_2D_data[indices, 0], reduced_2D_data[indices, 1], c=[cluster_colors[cluster]], label=f'Cluster {cluster}', s=10)

    # Plot the centroids with the same colors
    for cluster in range(10):
        ax.scatter(centroids_2d[cluster, 0], centroids_2d[cluster, 1], c=[cluster_colors[cluster]], marker='s', s=200, edgecolor='k')

    # Customize plot appearance
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(f'2D {method} Visualization of Reduced MNIST Data with K-means Clustering')
    ax.legend(title='Cluster')
    ax.grid(True)

    # Save the 2D plot for each method
    output_path_2d = f'./../results/PSI/{method}_2D_plot.png'
    plt.savefig(output_path_2d, bbox_inches='tight', dpi=300)

    # Show the 2D plot for each method
    plt.show()



        
        
	



	
