import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.sparse import load_npz


def loadpkl(filename):
    with open('Dataset_files/'+filename+'.pkl', 'rb') as f:
        dataset = pickle.load(f)
    return dataset

# g = loadpkl('graph_after_tconorm_but_before_phi_inv')
g = load_npz("Dataset_files/graph_before_inv.npz")
D_before_geod = loadpkl('D_before_geod')
D_after_geod = loadpkl('D')
# symmetrize
g = g + g.transpose() # symmetrize without factor of /2 because g is upper triangle matrix 
D_before_geod = D_before_geod + D_before_geod.transpose() # symmetrize without factor of /2 because D_before_geod is upper triangle matrix 
D_after_geod = (D_after_geod + D_after_geod.transpose())/2 # symmetrize
# convert sparse entries to infinities:
D_before_geod[D_before_geod==0] = 1e10
np.fill_diagonal(D_before_geod, 0) # except for diagonal

# cluster with linkage

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

def linkage_clustering(dists, method, clusternumber = 10, plot=False):
    # method can be "single", "average", and "complete".
    # Best results were obtained for "average"

    condensed_distances = squareform(dists) 
    # squareform makes distance matrix into a condensed distance matrix, which is required input to linkage clustering algo. A condensed distance matrix is a flat array containing the upper triangular of the distance matrix.
    Z = linkage(condensed_distances, method=method)

    if plot:
        fig = plt.figure(figsize=(25, 20))
        dn = dendrogram(Z)
        plt.show()

    cluster_labels = fcluster(Z, t=clusternumber, criterion='maxclust')

    return Z, cluster_labels

# Z, cluster_labels = linkage_clustering(D_after_geod, "average", clusternumber = 8, plot=False)

print("Linkage")

# clustering using Leiden
import leidenalg as la
import igraph as ig

def leiden_clustering(sparse_matrix, weighting_mode = 'undirected'):
    weighted_graph = ig.Graph.Weighted_Adjacency(sparse_matrix, mode=weighting_mode)
    # weighted_graph = ig.Graph.Weighted_Adjacency(sparse_matrix.tolist(), mode=weighting_mode)
    partition = la.find_partition(weighted_graph, la.ModularityVertexPartition)
    # partition = la.find_partition(weighted_graph, la.CPMVertexPartition)
    return partition

# leiden_partition = leiden_clustering(g)
# cluster_labels = np.array(leiden_partition.membership)

print("Leiden")

# comparison with MNIST dataset labels, groundtruth

# from data_and_plots import plot_data, load_MNIST
# import umap

# mnist_data = loadpkl('my_mnist')
mnist_labels = loadpkl('my_mnist_labels')

# umap_emb = umap.UMAP(random_state=42).fit_transform(mnist_data)

# plot_data(umap_emb, mnist_labels, title="umap with true labels", display=True, save=True)
# plot_data(umap_emb, cluster_labels, title="umap with cluster labels", display=True, save=True)


# cluster mds

from cluster_mds_old import cluster_mds

def phi_inv(x):
    return -np.log(x)

def cluster_algo(g):
    _, cluster_labels = linkage_clustering(g, "average", clusternumber = 10, plot=False)
    return cluster_labels

def cluster_algo2(g):
    leiden_partition = leiden_clustering(g)
    cluster_labels = np.array(leiden_partition.membership)
    return cluster_labels

# cluster_mds(D_after_geod, cluster_algo, true_labels=mnist_labels)
cluster_mds(g, cluster_algo2, geodesic=False, phi_inv=phi_inv, true_labels=mnist_labels, global_embedding=False)