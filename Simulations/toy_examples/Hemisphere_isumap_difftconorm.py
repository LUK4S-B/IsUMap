import os
import sys

# Set the path to the directory containing `isumap.py`
PATH_CURRENT = "../"  # Adjust this path as needed

scriptPath = os.path.abspath(PATH_CURRENT)
sys.path.append(scriptPath)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from sklearn.utils import check_random_state
from time import time
from isumap import isumap  

fig_dir = './../results/Hemisphere/'

# Set the seed for reproducibility
np.random.seed(0)

# Parameters
N = 10000

# Generate random data
rng = np.random.default_rng(seed=0)
theta = rng.uniform(0, 2 * np.pi, size=N)
phi = rng.uniform(0, 0.5 * np.pi, size=N)
t = rng.uniform(0, 1, size=N)
P = np.arccos(1 - 2 * t)

# Conditionally define x, y, z based on P
if ((P < (np.pi - (np.pi / 4))) & (P > ((np.pi / 4)))).all():
    indices = (P < (np.pi - (np.pi / 4))) & (P > ((np.pi / 4)))
    x, y, z = np.sin(P[indices]) * np.cos(theta[indices]), \
              np.sin(P[indices]) * np.sin(theta[indices]), \
              np.cos(P[indices])
else:
    x, y, z = np.sin(phi) * np.cos(theta), \
              np.sin(phi) * np.sin(theta), \
              np.cos(phi)

data = np.array([x, y, z]).T

# Plot the 3D scatter plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, s=2)
ax.view_init(20, -20)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Original Data')
plt.savefig(fig_dir + 'original_data.png')
plt.close()

# Parameters for isumap
t_conorms = ['canonical', 'probabilistic sum', 'bounded sum', 'drastic sum']
for tconorm in t_conorms:
    normalize = True
    distBeyondNN = False
    labels = None
    metricMDS = True

    # Run isumap
    reduced_data_canonical = isumap(data, k=30, d=2,
                                    normalize=normalize, distBeyondNN=distBeyondNN, verbose=True,
                                    dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
                                    labels=labels,
                                    initialization="cMDS", metricMDS=metricMDS, sgd_n_epochs=1000, sgd_lr=1e-2,
                                    sgd_batch_size=None,
                                    sgd_max_epochs_no_improvement=100,
                                    sgd_loss='MSE', sgd_saveplots_of_initializations=False, sgd_saveloss=False,
                                    tconorm=tconorm)
    reduced_isumap_c = reduced_data_canonical[0]

    # Plot the reduced data
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(reduced_isumap_c[:, 0], reduced_isumap_c[:, 1], s=2)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    #plt.title(f'Reduced Data using ISUMAP ({tconorm})')
    plt.savefig(fig_dir + f'reduced_data_isumap_{tconorm.replace(" ", "_")}.png')
    plt.close()
