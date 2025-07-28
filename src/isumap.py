import numpy as np

from time import time
from numba import njit, prange
from dimension_reduction_schemes import reduce_dim
from data_and_plots import printtime

from distance_graph_generation import distance_graph_generation

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

@njit(parallel=True)
def euclideanDistance(x, y):
    '''Computes euclidean distance of two arrays.'''
    diff_xy = np.subtract(x, y)
    return np.sqrt(np.dot(diff_xy, diff_xy))

@njit(parallel=True)
def euclideanDistanceMatrixOfArray(pointArray):
    nc = pointArray.shape[0]
    distanceMatrix = np.zeros((nc, nc))
    for i in prange(nc):
        for j in range(i + 1):
            if i == j:
                distanceMatrix[i, j] = 0.0
            else:
                distanceMatrix[i, j] = euclideanDistance(pointArray[i], pointArray[j])
                distanceMatrix[j, i] = distanceMatrix[i, j]
    return distanceMatrix

def compute_mean_points_and_labels(nc, data, SM):
    '''
    Computes cluster centers for a collection of submatrices of the distance matrix corresponding to connected components.

    :param nc: number of clusters/connected components
    :param data: data matrix of shape (n,m)
    :param SM: list of tuples (distance_matrices,indices)
    :return meanPointsOfClusters: mean points of the clusters
    :return clusterLabels: cluster labels
    '''
    meanPointsOfClusters = np.empty((nc, data[0].size))
    clusterLabels = []
    for i in prange(nc):
        meanPointsOfClusters[i] = data[SM[i][1]].mean(0)
        clusterLabels.append(float(i) * np.ones(len(SM[i][1])))
    clusterLabels = np.concatenate(clusterLabels)
    return meanPointsOfClusters, clusterLabels

def subMatrixEmbeddings(nc, SM, meanPointEmbeddings, **reduce_dim_kwargs):

    '''
    applys function reduce_dim to a list of submatrices, corresponding to connected components


    :param nc: nc: number of clusters/connected components
    :param SM: list of tuples (distance_matrices,indices)
    :param meanPointEmbeddings: embeddings of mean points of connected components
    :param reduce_dim_kwargs: arguments for reduce_dim function
    :return:
    '''
    submatrixEmbeddings = []
    submatrixInitEmbeddings = []
    for i in range(nc):
        subMatrixInitEmb, subMatrixEmb = reduce_dim(SM[i][0], **reduce_dim_kwargs)

        submatrixInitEmbeddings.append(subMatrixInitEmb)
        submatrixEmbeddings.append(subMatrixEmb)

        submatrixInitEmbeddings[i] += meanPointEmbeddings[i]  # adds the meanPointEmbeddings[i]-vector to each row of the subMatrixInitEmbeddings[i]-matrix
        submatrixEmbeddings[i] += meanPointEmbeddings[i]  # adds the meanPointEmbeddings[i]-vector to each row of the subMatrixEmbeddings[i]-matrix

    return submatrixInitEmbeddings, submatrixEmbeddings


def isumap(data,
           k: int,
           d: int,
           normalize:bool = True,
           distBeyondNN:bool = True,
           verbose: bool = True,
           dataIsDistMatrix: bool = False,
           dataIsGeodesicDistMatrix: bool = False,
           saveDistMatrix: bool = False,
           initialization = "cMDS",
           metricMDS : bool = True,
           sgd_n_epochs:int = 1000,
           sgd_lr: float = 1e-2,
           sgd_batch_size = None,
           sgd_max_epochs_no_improvement: int = 100,
           sgd_loss = 'MSE',
           sgd_saveloss: bool = False,
           tconorm = "canonical",
           distFun = "canonical",
           phi = None,
           phi_inv = None,
           epm = True,
           m_scheme_value = 1.0,
           apply_Dijkstra = True,
           extractSubgraphs = True,
           max_param = 100.0,
           **phi_params):

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

    N = data.shape[0]
    D = distance_graph_generation(data,
                                    k,
                                    normalize,
                                    distBeyondNN,
                                    verbose,
                                    dataIsDistMatrix,
                                    dataIsGeodesicDistMatrix,
                                    saveDistMatrix,
                                    tconorm,
                                    distFun,
                                    phi,
                                    phi_inv,
                                    epm,
                                    m_scheme_value,
                                    apply_Dijkstra,
                                    max_param,
                                    **phi_params)

    t0 = time()
    if extractSubgraphs:
        SM = extractSubmatrices(D)
    else:
        S = np.array([i for i in range(N)])
        SM = [(D, S)]
    t1 = time()
    if verbose:
        printtime("Extracted connected components in",t1-t0)
    
    nc = len(SM)
    if verbose:
        print("Number of clusters = "+str(nc))

    t0 = time()
    meanPointsOfClusters, clusterLabels = compute_mean_points_and_labels(nc, data, SM)
    t1 = time()
    if verbose:
        printtime("Mean points and labels",t1-t0)
    
    if meanPointsOfClusters.shape[1] == d:
        meanPointEmbeddings = meanPointsOfClusters
    else:
        t0 = time()
        clusterDistanceMatrix = euclideanDistanceMatrixOfArray(meanPointsOfClusters)
        t1 = time()
        if verbose:
            printtime("Euclidean distances",t1-t0)
        
        t0 = time()
        _, meanPointEmbeddings = reduce_dim(clusterDistanceMatrix, d=d, n_epochs = sgd_n_epochs, lr=sgd_lr, batch_size = sgd_batch_size, max_epochs_no_improvement = sgd_max_epochs_no_improvement, loss = sgd_loss, initialization="cMDS", metricMDS=False, saveloss=False, verbose=verbose)
        t1 = time()
        if verbose:
            printtime("Embedded the cluster mean points in",t1-t0)
    
    if verbose:
        print("\nReducing dimension...")
    t0 = time()
    submatrixInitEmbeddings, submatrixEmbeddings = subMatrixEmbeddings(nc, SM, meanPointEmbeddings, d=d, n_epochs=sgd_n_epochs, lr=sgd_lr, batch_size=sgd_batch_size, max_epochs_no_improvement=sgd_max_epochs_no_improvement, loss=sgd_loss, initialization=initialization, metricMDS=metricMDS, saveloss=sgd_saveloss, verbose=verbose)

    t1 = time()
    if verbose:
        printtime("Computed submatrix embeddings in",t1-t0)
    
    finalInitEmbedding = np.concatenate(submatrixInitEmbeddings)
    finalEmbedding = np.concatenate(submatrixEmbeddings)
    
    return  finalInitEmbedding, finalEmbedding, clusterLabels

