import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_CURRENT = os.path.join(SCRIPT_DIR, "../../src/")
scriptPath = os.path.abspath(PATH_CURRENT)
sys.path.append(scriptPath)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from mpl_toolkits.mplot3d import Axes3D
from isumap import isumap  
from sklearn.utils import check_random_state
from sklearn import datasets

# Generate a Möbius strip
def create_nonuniform_mobius(n_samples, width):
    rng = np.random.default_rng(seed=0)
    u = rng.uniform(0, 2 * np.pi, size=n_samples)
    v = rng.beta(1, 2, size=n_samples) * width * 2 - width
    x = (1 + v * np.cos(u / 2)) * np.cos(u)
    y = (1 + v * np.cos(u / 2)) * np.sin(u)
    z = v * np.sin(u / 2)
    data = np.array([x, y, z]).T
    return data, u

# Fetch MNIST dataset
def fetch_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'] / 255.0  # Normalize pixel values
    y = mnist['target'].astype(int)
    return X[:10000], y[:10000]  # Use only a subset of MNIST for performance

# Parameters
n = 10000  # Number of samples
datasets_list = [
    ('Swiss Roll with Hole', lambda: datasets.make_swiss_roll(n_samples=3000, hole=True, noise=0.0, random_state=0)),
    ('Möbius', lambda: create_nonuniform_mobius(n_samples=3000, width=1)),
    ('MNIST', fetch_mnist)
]

phi_list = ['exp', 'half_normal', 'log_normal', 'pareto', 'uniform']  # Different phi functions

# Create subplots with rows = len(datasets_list) and columns = len(phi_list) + 1
fig, axes = plt.subplots(len(datasets_list), len(phi_list) + 1, figsize=(15, 15))

for row_idx, (dataset_name, dataset_func) in enumerate(datasets_list):  
    # Generate dataset
    X, t = dataset_func()

    # Plot the original dataset in the first column
    if dataset_name in ['Swiss Roll with Hole']:  
        ax = fig.add_subplot(len(datasets_list), len(phi_list) + 1, row_idx * (len(phi_list) + 1) + 1, projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=25, c=t, cmap=plt.cm.rainbow)
        ax.view_init(10, 120)
        ax.set_title(f'a', fontsize=10)

    elif dataset_name in ['Möbius']:
    	ax = fig.add_subplot(len(datasets_list), len(phi_list) + 1, row_idx * (len(phi_list) + 1) + 1, projection='3d')
    	ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=25, c=t, cmap=plt.cm.rainbow)
    	ax.view_init(17,40)
    	ax.set_title(f'a') 
    	
    else:  # 2D dataset (MNIST)
        ax = axes[row_idx, 0]
        scatter = ax.scatter(X[:, 0], X[:, 1], s=25, c=t, cmap=plt.cm.rainbow)
        ax.set_title(f'a', fontsize=10)

   
    for col_idx, current_phi in enumerate(phi_list):
        normalize = True
        distBeyondNN = True
        labels = t
        metricMDS = True

        try:
            # Run ISUMAP
            reduced_data_isumap = isumap(
                X, k=30, d=2, normalize=normalize, distBeyondNN=distBeyondNN, verbose=False,
                dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
                labels=labels, initialization="cMDS", metricMDS=metricMDS,
                sgd_n_epochs=1500, sgd_lr=1e-2, sgd_batch_size=None,
                sgd_max_epochs_no_improvement=75, sgd_loss='MSE',
                sgd_saveplots_of_initializations=False, sgd_saveloss=False,
                tconorm='drastic sum', phi=current_phi
            )
            reduced_isumap_c = reduced_data_isumap[0]

            
            ax = axes[row_idx, col_idx + 1]
            scatter = ax.scatter(reduced_isumap_c[:, 0], reduced_isumap_c[:, 1], s=25, c=t, cmap=plt.cm.rainbow)
            ax.set_title(f'phi={current_phi}', fontsize=10)
        except ValueError as e:
            # Skip invalid phi values and report the error
            ax = axes[row_idx, col_idx + 1]
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', fontsize=8, wrap=True)
            ax.set_title(f'phi={current_phi}', fontsize=10)


plt.tight_layout()

# Save the figure
fig_dir = './../results/Diff_phi'  # Update this to a valid directory
import os
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(fig_dir + 'reduced_data_isumap_diff_phi_drastic_sum.png')


