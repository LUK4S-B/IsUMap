import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn import manifold, datasets
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.utils import check_random_state
from time import time
from isumap import isumap  

fig_dir = '/user/people/fahimi/Documents/IsUMAP_diff_conorm/isumap_diff_t-conorm_results/isumap_hemisphere/'

# Generate random data
def make_swiss_roll(n_samples=1000, s=1, noise=0.0, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random points in the latent space
    t = np.random.rand(n_samples) * s * np.pi
    height = np.random.rand(n_samples) * 60

    # Generate the Swiss roll dataset in 3D
    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)

    # Add noise
    x += noise * np.random.randn(n_samples)
    y += noise * np.random.randn(n_samples)
    z += noise * np.random.randn(n_samples)

    return np.column_stack((x, y, z)), t

# Parameters
S = [ 8, 10, 12]
for i in S:
    samples = 10000
    noise = 0.0
    random_state = 0

    # Generate Swiss roll
    X, t = make_swiss_roll(n_samples=samples, s=i, noise=noise, random_state=random_state)

    # Plot the 3D scatter plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=t, cmap=plt.cm.rainbow)
    ax.view_init(10, -70)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Original Data (s={i})')
    plt.colorbar(sc)
    plt.savefig(fig_dir + f'original_data_{i}.png')
    plt.close()

    # Parameters for isumap
    NN = [10, 20, 30, 50, 100, 200]
    for n in NN:
        normalize = True
        distBeyondNN = True
        labels = t
        metricMDS = True

        # Run isumap
        reduced_data_canonical = isumap(X, k=n, d=2,
                                        normalize=normalize, distBeyondNN=distBeyondNN, verbose=True,
                                        dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
                                        labels=labels,
                                        initialization="cMDS", metricMDS=metricMDS, sgd_n_epochs=1000, sgd_lr=1e-2,
                                        sgd_batch_size=None,
                                        sgd_max_epochs_no_improvement=100,
                                        sgd_loss='MSE', sgd_saveplots_of_initializations=False, sgd_saveloss=False,
                                        tconorm='canonical')
        reduced_isumap_c = reduced_data_canonical[0]

        # Plot the reduced data
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(reduced_isumap_c[:, 0], reduced_isumap_c[:, 1], s=2,c=t, cmap=plt.cm.rainbow)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        #plt.title(f'Reduced Data using ISUMAP (s={i}, NN={n})')
        plt.savefig(fig_dir + f'reduced_data_isumap_{i}_{n}.png')
        plt.close()


	
