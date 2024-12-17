import os
import sys

# Set the path to the directory containing isumap.py
PATH_CURRENT = "../"  # Adjust this path as needed

scriptPath = os.path.abspath(PATH_CURRENT)
sys.path.append(scriptPath)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import check_random_state
from sklearn import datasets, manifold
from isumap import isumap
import umap

# Directory for saving figures
Figure_dir = './../results/SwissRollwithHolew_Mobius'

# Parameters
n_samples = 3000  # Number of samples for datasets
n_neighbors = 20  # Number of neighbors for UMAP and IsUMap
random_state = check_random_state(0)

# Load datasets
def creat_nonuniform_Mobius(n_samples , width):
	rng = np.random.default_rng(seed=0)
	u = rng.uniform(0, 2 * np.pi, size=n_samples) 
	v = rng.beta(1, 2, size=n_samples) * width * 2 - width 
	color = u
	x = (1 + v * np.cos(u / 2)) * np.cos(u)
	y = (1 + v * np.cos(u / 2)) * np.sin(u)
	z = v * np.sin(u / 2)
	return np.array([x, y, z]).T, color
	

datasets_list = {
    'Swiss Roll with Hole': lambda: datasets.make_swiss_roll(n_samples=n_samples, hole = True, noise=0.0, random_state=random_state),
    'Mobius': lambda: creat_nonuniform_Mobius(n_samples = n_samples, width =1),
    
}

# Prepare the figure (4 columns: Original, ISUMAP, Isomap, UMAP)
fig, axes = plt.subplots(len(datasets_list), 4, figsize=(30, 10 * len(datasets_list)))

# Parameters for ISUMAP
normalize = True
distBeyondNN = False
metricMDS = True

# Create the datasets and their visualizations
for i, (title, data_func) in enumerate(datasets_list.items()):
    # Get the dataset
    data, labels = data_func()
    
    
    # Original data plot
    if title in ['Swiss Roll with Hole']:
    	ax = fig.add_subplot(len(datasets_list), 4, i * 4 + 1, projection='3d')
    	ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=25, c=labels, cmap='rainbow')
    	ax.view_init(10,120)
    	ax.set_title(f'a')
    else:
    	ax = fig.add_subplot(len(datasets_list), 4, i * 4 + 1, projection='3d')
    	ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=25, c=labels, cmap='rainbow')
    	ax.view_init(17,40)
    	ax.set_title(f'a')

    # ISUMAP
    normalize = True
    distBeyondNN = False
    metricMDS = True
    
    reduced_data_isumap = isumap(data, k=n_neighbors, d=2,
                                    normalize=normalize, distBeyondNN=distBeyondNN, verbose=True,
                                    dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
                                    labels=labels,
                                    initialization="cMDS", metricMDS=metricMDS, sgd_n_epochs=1000, sgd_lr=1e-2,
                                    sgd_batch_size=None,
                                    sgd_max_epochs_no_improvement=100,
                                    sgd_loss='MSE', sgd_saveplots_of_initializations=False, sgd_saveloss=False,
                                    tconorm='canonical')

    reduced_isumap_c = reduced_data_isumap[0]
    ax = axes[i, 1]  
    ax.scatter(reduced_isumap_c[:, 0], reduced_isumap_c[:, 1], s=25, c=labels, cmap='rainbow')
    ax.set_title(f'b')

    # Isomap
    reduced_data_isomap = manifold.Isomap(n_components=2, n_neighbors=n_neighbors).fit_transform(data)
    ax = axes[i, 2]  
    ax.scatter(reduced_data_isomap[:, 0], reduced_data_isomap[:, 1], s=25, c=labels, cmap='rainbow')
    ax.set_title(f'c')

    # UMAP
    reduced_umap = umap.UMAP(n_components=2, n_neighbors=n_neighbors).fit_transform(data)
    ax = axes[i, 3]  
    ax.scatter(reduced_umap[:, 0], reduced_umap[:, 1], s=25, c=labels, cmap='rainbow')
    ax.set_title(f'd')
    


plt.tight_layout()
os.makedirs(Figure_dir, exist_ok=True) 
output_file = os.path.join(Figure_dir, 'SwissRoll_Mobius.png')
plt.savefig(output_file, dpi=300)
plt.show()











