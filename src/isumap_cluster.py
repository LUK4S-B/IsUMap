import numpy as np

from time import time
from numba import njit, prange
from dimension_reduction_schemes import reduce_dim
from data_and_plots import printtime

from distance_graph_generation import distance_graph_generation

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_CURRENT = os.path.join(SCRIPT_DIR, "./cluster_mds/")
scriptPath = os.path.abspath(PATH_CURRENT)
sys.path.append(scriptPath)
from cluster_mds import cluster_mds
from cluster_algos import linkage_cluster, linkage_clustering #, leiden_cluster, leiden_clustering

@njit
def extractSubmatrices(D):
    '''
    Extracts submatrices of connected components of a distance matrix.
    The distance matrix has to have np.inf whenever two points are in different connected components,
    and this property needs to be transitive.

    :param D: np.ndarray(n,n) - distance matrix with np.inf indicating points being in different connected components
    :return Sms: list of  tuples (np.ndarray,indices) - distance matrices of the component together with indices
    '''
    N = D.shape[0]
    S = np.array([i for i in range(N)])
    SMs = []
    i = 0
    while S.size != 0:
        indSet = np.where(D[S[0]] != np.inf)[0]
        n = len(indSet)
        SMs.append((np.empty((n, n)), indSet))
        for j in range(n):
            SMs[i][0][j] = D[indSet[j]][indSet]
        # Manual implementation of np.setdiff1d
        mask = np.ones(S.shape[0], dtype=np.bool_)
        for ind in indSet:
            mask[np.where(S == ind)] = False
        S = S[mask]
        i += 1
    return SMs


def isumap_cluster(data,
           k: int,
           normalize:bool = True,
           distBeyondNN:bool = True,
           verbose: bool = True,
           dataIsDistMatrix: bool = False,
           dataIsGeodesicDistMatrix: bool = False,
           saveDistMatrix: bool = False,
           saveDistMatrixPath = None,
           tconorm = "canonical",
           distFun = "canonical",
           phi = None,
           phi_inv = None,
           epm = True,
           m_scheme_value = 1.0,
           save_fuzzy_graph = False,
           apply_Dijkstra = True,
           extractSubgraphs = True,
           max_param = np.inf,
           cluster_algo = "linkage_cluster",
           labels = None,
           return_fuzzy_graph = False,
           global_embedding = True,
           directedDistances = False,
           store_results = False,
           display_results = False,
           save_display_results = False,
           plot_title = "Title",
           also_return_optimizer_model = False,
           also_return_medoid_paths = False,
           preprocess_with_pca = False,
           pca_components = 40,
           plot_original_data = False,
           visualize_results = False,
           custom_color_map = 'jet',
           **kwargs):

    '''

    IsUMap embeds data into a d-dimensional space, using the colimit in the category of uber-metric spaces, by means
    of Dijkstra's algorithm after a merging of local metric spaces via the minimum.
    The low-dimensional points are then obtained via multidimensional scaling.


    Distances are normalized to obtain a manifold where the data is uniformly distributed as in UMAP (this step is optional,
    else, the method reduces to ISOMAP).
    Connected components of the data are embedded separately, centered around their means.



    :param data: np.ndarray (n,m) - matrix of data points
    :param k: int - number of nearest neighbors to use in distance computation
    :param d: int - dimensionality of the embedding
    :param normalize: bool - if True, data is normalized by k-nn distance
    :param distBeyondNN: bool - if True, distance of nearest neighbor is subtracted from distances
    :param verbose: bool - if True, prints progress and time information
    :param dataIsDistMatrix: bool - whether data is a distance matrix, if False, distance is computed from data
    :param dataIsGeodesicDistMatrix: bool - whether data is geodesic distance matrix, if False, dijkstra is applied
    :param saveDistMatrix:  bool -whether to save distance matrix
    :param initialization: initialization for dimensionality reduction
    :param metricMDS: bool - if True, metricMDS is used for dimensionality reduction
    :param sgd_n_epochs: int - number of epochs for SGD, only used if metricMDS = True
    :param sgd_lr: float -lr for SGD, only used if metricMDS = True
    :param sgd_batch_size: int - batch size for SGD, only used if metricMDS=True
    :param sgd_max_epochs_no_improvement: int - number of epochs until early stopping, only used if metricMDS = True
    :param sgd_loss: loss for SGD (one of ['MSE','Sammon'] or custom loss)
    :param sgd_saveloss:  bool - whether to save plots of SGD loss
    :param tconorm: the t-conorm used for symmetrization - one of ['canonical', 'probabilistic sum', 'bounded sum', 'drastic sum', 'Einstein sum', 'nilpotent maximum']
    :param phi: None or str or callable: phi function to transfer metrics to fuzzy weights. If None, defaults to exponential function.
    If a string, must be one of ['exp','half_normal','log_normal','pareto','uniform']. Else, custom function may be used. Has to be callable, and an inverse phi_inv has to be provided.
    Does not do anything if t-conorm is 'canonical'.
    :param phi_inv: None or callable: inverse of phi function. If None, defaults to log. Else, must be callable inverse of phi.
    :param epm: If this parameter is set to True, then all non-radial distances in the star graph are set to Infinity as if we were in the category EPMet. Default is False.
    :param **phi_params**: additional parameters to pass to the phi function.
    :return finalInitEmbedding: np.ndarray (n,d) - initial low dimensional embedding before the application of metric MDS 
    :return finalEmbedding: np.ndarray (n,d) - low dimensional embedding. If metricMDS = False, then finalEmbedding==finalInitEmbedding.
    :return clusterLabels: labels of connected components of the distance matrix

    ...............................................

    Example Usage:

    X = np.random.normal((1000,10))
    Y = isumap(X,k=15,d=2)


    '''

    phi_kwargs = kwargs.get('phi_kwargs') or {}
    cluster_algo_kwargs = kwargs.get('cluster_algo_kwargs') or {}


    if cluster_algo == "linkage_cluster":
        cluster_algo = linkage_cluster
    # elif cluster_algo == "leiden_cluster":
    #     cluster_algo = leiden_cluster
    #     return_fuzzy_graph = True # Leiden clustering must be appplied to a fuzzy graph
    #     global_embedding = False

    N = data.shape[0]
    D, phi_inv, data = distance_graph_generation(data,
                                    k,
                                    normalize = normalize,
                                    distBeyondNN = distBeyondNN,
                                    verbose = verbose,
                                    dataIsDistMatrix = dataIsDistMatrix,
                                    dataIsGeodesicDistMatrix = dataIsGeodesicDistMatrix,
                                    saveDistMatrix = saveDistMatrix,
                                    saveDistMatrixPath = saveDistMatrixPath,
                                    tconorm = tconorm,
                                    distFun = distFun,
                                    phi = phi,
                                    phi_inv = phi_inv,
                                    epm = epm,
                                    m_scheme_value = m_scheme_value,
                                    save_fuzzy_graph = save_fuzzy_graph,
                                    apply_Dijkstra = apply_Dijkstra,
                                    max_param = max_param,
                                    return_fuzzy_graph = return_fuzzy_graph,
                                    preprocess_with_pca = preprocess_with_pca,
                                    pca_components = pca_components,
                                    **phi_kwargs)

    # t0 = time()
    # if extractSubgraphs:
    #     SM = extractSubmatrices(D)
    # else:
    #     S = np.array([i for i in range(N)])
    #     SM = [(D, S)]
    # t1 = time()
    # if verbose:
    #     printtime("Extracted connected components in",t1-t0)
    
    # nc = len(SM)
    # if verbose:
    #     print("Number of connected components = "+str(nc))

    # if nc > 1:
    #     print("Number of connected components > 1. This is not yet implemented.") # TODO: implement
    #     return None
    
    geodesic = not return_fuzzy_graph and apply_Dijkstra

    results = cluster_mds(D, cluster_algo, geodesic = geodesic, verbose = verbose, phi_inv = phi_inv, true_labels = labels, global_embedding = global_embedding, directedDistances = directedDistances, store_results = store_results, display_results = display_results, save_display_results = save_display_results, plot_title = plot_title, also_return_optimizer_model = also_return_optimizer_model, also_return_medoid_paths = also_return_medoid_paths, orig_data = data, plot_original_data = plot_original_data, visualize_results = visualize_results, custom_color_map=custom_color_map, **cluster_algo_kwargs)


    return results

