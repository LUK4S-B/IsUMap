from utils import *
import matplotlib.pyplot as plt
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from persim import plot_diagrams
from scipy.spatial.distance import pdist
import umap
import os

# Set directory and parameters
fig_dir = "./../results/Grid_cell/"
rat_name = 'R'
mod_name = '1'
sess_name = 'OF'
day_name = 'day1'

# Parameters for analysis
bRoll = False
dim = 6
ph_classes = [0, 1]
num_circ = len(ph_classes)
dec_tresh = 0.99
metric = 'cosine'
maxdim = 1
coeff = 47
active_times = 15000
k = 1000
num_times = 5
n_points = 1200
nbs = 800
sigma = 1500
folder = 'https://figshare.com/articles/dataset/Toroidal_topology_of_population_activity_in_grid_cells/16764508?file=35078602'

def neg_exp(t):
    return np.exp(-t)

def umap_weight(distances, k, f=neg_exp, alpha=0.1):
    p = np.min(distances, axis=1, initial=np.inf, where=distances > 0)
    whereinf = np.isinf(p)
    if whereinf.sum() > 0:
        p[whereinf] = 0
        print("Warning... There are point(s) in the dataset where all nearest neighbors are the same point.")
        print(p)

    N = distances.shape[0]
    weights = np.empty_like(distances)
    new_distances = np.empty_like(distances)
    for i in range(N):
        weight, new_distance = sigmasearch(distances[i], k, p[i], f=f, alpha=alpha)
        weights[i] = weight
        new_distances[i] = new_distance
    
    return weights, new_distances

def sigmasearch(distances, k, rho, tolerance=1e-5, max_iter=64, f=neg_exp, alpha=1.0):
    target = np.log2(k)
    high = np.inf
    mid = 1.0
    lo = 0.0
    counter = 0
    weightsum = np.inf
    while np.abs(target - weightsum) > tolerance and counter <= max_iter:
        sigma = mid
        dist = np.maximum((distances - alpha * rho) / sigma, 0)
        z = f(dist)
        weightsum = np.sum(z)
        if weightsum > target:
            high = mid
            mid = (lo + high) / 2.0
        else:
            lo = mid
            if high == np.inf:
                mid *= 2
            else:
                mid = (lo + high) / 2.0
        counter += 1
    return z, dist

def union_fuzzy_simplicial_set(W, set_op_mix_ratio=1.0, norm=1):
    symmetric_weight = np.zeros(W.shape, dtype=np.float32)
    transpose = W.transpose()
    
    if not norm in range(4):
        raise ValueError("Invalid t-conorm type")
    elif norm == 0:
        symmetric_weight = np.maximum(W, transpose)
    elif norm == 1:
        prod_matrix = np.multiply(W, transpose)
        symmetric_weight = set_op_mix_ratio * (W + transpose - prod_matrix) + (1.0 - set_op_mix_ratio) * prod_matrix
    elif norm == 2:
        symmetric_weight = np.minimum(W + transpose, 1)
    elif norm == 3:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if W[i][j] == 0:
                    symmetric_weight[i][j] = transpose[i][j]
                elif transpose[i][j] == 0:
                    symmetric_weight[i][j] = W[i][j]
                else:
                    symmetric_weight[i][j] = 1
    
    return symmetric_weight

def data_dist(X, k, d, norm=1, min_dist='auto', f=neg_exp, alpha='auto', apply_set_operations=True):
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(X)
    knn_distance, knn_inds = nn.kneighbors(X)
    
    weight, distance = umap_weight(knn_distance, k, f=f, alpha=1.0)

    R = np.zeros((X.shape[0], X.shape[0]))
  
    for i in range(knn_inds.shape[0]):
        R[i, knn_inds[i]] = np.maximum(R[i, knn_inds[i]], weight[i]) 
        
    if apply_set_operations:
        UMAP_sym_weight = union_fuzzy_simplicial_set(R, norm=norm)
        data_D = -np.log(UMAP_sym_weight)
        return data_D
    else:
        return None

# Main processing loop
for rat_name, mod_name, sess_name, day_name in [('R', '1', 'OF', 'day1')]:
    sspikes = get_spikes(rat_name, mod_name, day_name, sess_name, bType='pure', bSmooth=True, bSpeed=True, folder=folder)[0]
    num_neurons = len(sspikes[0, :])

    if bRoll:
        np.random.seed(42)
        shift = np.zeros(num_neurons, dtype=int)
        for n in range(num_neurons):
            shifti = int(np.random.rand() * len(sspikes[:, 0]))
            sspikes[:, n] = np.roll(sspikes[:, n].copy(), shifti)
            shift[n] = shifti
            
    times_cube = np.arange(0, len(sspikes[:, 0]), num_times)
    movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube, :], 1))[-active_times:])
    movetimes = times_cube[movetimes]
    
    dim_red_spikes_move_scaled,__,__ = pca(preprocessing.scale(sspikes[movetimes,:]), dim = dim)
    indstemp,dd,fs  = sample_denoising(dim_red_spikes_move_scaled,  k, 
                                       n_points, 1, metric)
    dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]

    #pre_red_spikes_move = preprocessing.scale(sspikes[movetimes, :])
    #print('pre_red_spikes_move', pre_red_spikes_move.shape)
    #indstemp, dd, fs = sample_denoising(pre_red_spikes_move, k, n_points, 1, metric)
    #pre_red_spikes_move = pre_red_spikes_move[indstemp, :]

    # Run UMAP
    reduced_umap = umap.UMAP(n_neighbors=15, n_components=dim, metric=metric, random_state=42)
    reduced_data = reduced_umap.fit_transform(dim_red_spikes_move_scaled)

    # Plot 3D embedding
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], s=5)
    ax.view_init(azim=100, elev=320)

    # Save the 3D embedding plot
    plot_path = os.path.join(fig_dir, f"{rat_name}_{mod_name}_{sess_name}_{day_name}_3D_embedding.png")
    plt.savefig(plot_path)
    plt.show()  # Show plot in interactive mode
    plt.close()

    # Calculate distance matrix based on UMAP symmetric weight
    #dist = pdist(reduced_data , metric='euclidean')
    #print(dist.shape)
    #dist = data_dist(X=pre_red_spikes_move, k=nbs, d=3, norm=1, min_dist='auto', f=neg_exp, alpha='auto', apply_set_operations=True)
    

    # Calculate persistence using Ripser
    persistence = ripser(reduced_data, maxdim=maxdim, coeff=coeff, do_cocycles=True, distance_matrix=False, metric=metric)

    # Plot and save persistence diagrams
    persistence_path = os.path.join(fig_dir, f"{rat_name}_{mod_name}_{sess_name}_{day_name}_persistence_diagrams.png")
    plot_diagrams(persistence['dgms'])
    plt.savefig(persistence_path)
    plt.close()

    # Plot and save barcodes
    barcode_path = os.path.join(fig_dir, f"{rat_name}_{mod_name}_{sess_name}_{day_name}_barcode.png")
    plot_barcode(persistence['dgms'])
    plt.savefig(barcode_path)
    plt.close()

    if len(day_name) > 0:
        day_name = '_' + day_name
    print(f"{rat_name}_{mod_name}_{sess_name}{day_name}")

plt.show()  # Show all figures if running interactively

