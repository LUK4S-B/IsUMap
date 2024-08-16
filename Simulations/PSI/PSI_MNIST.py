import numpy as np
import pandas as pd
from sklearn.cluster
import KMeans
from genieclust 
import compare_partitions
import seaborn as sns
import matplotlib.pyplot as plt

# Directories for each method
save_dirs = {
    'IsUMAP': '/usr/people/fahimi/Documents/IsUMAP_diff_conorm/results/MNIST_isumap_csv',
    'Isomap': '/usr/people/fahimi/Documents/IsUMAP_diff_conorm/results/MNIST_isomap_csv',
    'UMAP': '/usr/people/fahimi/Documents/IsUMAP_diff_conorm/results/MNIST_umap_csv'
}

# Reduced dimensions to evaluate
D = [400, 300, 200, 150, 100, 50, 30, 10, 5, 2]

# Initialize a DataFrame to store PSI values for all methods
psi_data = pd.DataFrame()

for method, save_dir in save_dirs.items():
    psi_values = []
    
    for d in D:
        # Load reduced data and labels
        reduced_data = np.loadtxt(f'{save_dir}/reduced_umap_d_{d}.csv', delimiter=',')
        reduced_labels = pd.read_csv(f'{save_dir}/reduced_umap_labels_d_{d}.csv')['label'].values
        
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

# Set the x-axis ticks explicitly
plt.xticks(D)

# Save the plot
output_path_psi_combined = '/usr/people/fahimi/Documents/IsUMAP_diff_conorm/results/PSI_combined_plot.png'
plt.savefig(output_path_psi_combined, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()




        
        
	



	
