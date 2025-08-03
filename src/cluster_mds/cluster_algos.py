import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

import leidenalg as la
import igraph as ig


def leiden_clustering(sparse_matrix, weighting_mode = 'undirected'):
    weighted_graph = ig.Graph.Weighted_Adjacency(sparse_matrix, mode=weighting_mode)
    partition = la.find_partition(weighted_graph, la.ModularityVertexPartition)
    # partition = la.find_partition(weighted_graph, la.CPMVertexPartition)
    return partition

def leiden_cluster(g, weighting_mode = 'undirected'):
    leiden_partition = leiden_clustering(g, weighting_mode = weighting_mode)
    cluster_labels = np.array(leiden_partition.membership)
    return cluster_labels

def linkage_clustering(dists, method = "average", clusternumber = 10, plot=False):
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

def linkage_cluster(g, clusternumber = 10, plot=False):
    _, cluster_labels = linkage_clustering(g, method = "average", clusternumber = clusternumber, plot=plot)
    return cluster_labels
