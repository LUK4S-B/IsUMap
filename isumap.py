import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors
from time import time
import pickle
from numba import njit, prange
from multiprocessing import Pool, cpu_count
from functools import partial
from dimensionReductionSchemes import reduce_dim
from data_and_plots import printtime
import torch
import warnings

eps = np.finfo(np.float32).tiny

IMPLEMENTED_PHIS = ['exp','half_normal','log_normal','pareto','uniform']
def dijkstra_wrapper(graph, i):
    return dijkstra(csgraph=graph, indices=i)

@njit(parallel=True)
def find_nn_DistMatrix(M, k:int):
    r'''
    Implements an exact k nearest neighbour search.

    :param M: np.ndarray(n,n) - distance matrix
    :param k: int - number of nearest neighbors
    :return knn_inds: np.ndarray(n,k) - indices of nearest neighbors
    :return knn_distances: np.ndarray(n,d) - distances of nearest neighbors

    '''
    N = M.shape[0]
    knn_distances = np.empty((N, k))
    knn_inds = np.empty((N, k), dtype=np.int64)
    for i in prange(N):
        partitioned_indices = np.argpartition(M[i], k)
        knn_inds[i] = partitioned_indices[:k]
        knn_distances[i] = M[i][knn_inds[i]]
    return knn_inds, knn_distances

def find_nn(data, k):
    '''
    wrapper for nearest neighbour search via the scikitlearn implementation
    '''
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(data)
    knn_distances, knn_inds = nn.kneighbors(data)
    return knn_inds, knn_distances

@njit
def two_smallest(numbers):
    '''
    Computes the index of the minimum and the value of the second smallest element of a set.

    :param numbers: iterable of numeric
    :return ind_m1: index of minimum
    :return m2: second smallest number
    '''
    m1, m2 = np.inf, np.inf
    ind_m1 = 0
    i = 0
    for x in numbers:
        if x <= m1:
            m1, m2 = x, m1
            ind_m1 = i
        elif x < m2:
            m2 = x
        i = i + 1
    return ind_m1, m2

@njit(parallel=True)
def normalization(distances, normalize:bool, distBeyondNN:bool):
    r'''
    Normalizes a matrix of distances $D_{ij}$ to obtain normalized distances of the form

    $$\frac{D_{ij} - \rho_i}{\sigma_i}$$

    where $\rho_i$ is distance to the nearest neighbour nad $\sigma_i$ is distance to the furthest neighbor


    :param distances: np.ndarray(n,n) - distance matrix
    :param normalize:  bool - whether to perform division by furthest distance
    :param distBeyondNN: bool - whether to subtract nearest distance
    :return:
    '''
    if distBeyondNN == True and normalize == True:
        for i in prange(distances.shape[0]):
            ind_m1, m2 = two_smallest(distances[i])
            distances[i] = distances[i] - m2
            distances[i][ind_m1] = 0.0
            mdi = np.max(distances[i])
            if 1e-10 < mdi:
                distances[i] = distances[i] / mdi
    elif distBeyondNN == True and normalize == False:
        for i in prange(distances.shape[0]):

            #it might be possible to simplify the computations below if distances are already ordered
            ind_m1, m2 = two_smallest(distances[i])
            distances[i] = distances[i] - m2+eps
            distances[i][ind_m1] = 0.0
    elif distBeyondNN == False and normalize == True:
        for i in prange(distances.shape[0]):
            mdi = np.max(distances[i])
            if 1e-10 < mdi:
                distances[i] = distances[i] / mdi
    return distances

@njit(parallel=True)
def compR(knn_inds, distance):
    r'''
    Constructs a full $n\times n$ distance matrix from $(n,k)$ nearest neighbor distances, where non-neighbor entries
    are filled with $0$.

    :param knn_inds: np.ndarray(n,k) - indices of k-nearest neighbors
    :param distance: np.ndarray(n,k) - ditances to k-nearest neighbors
    :return R: np.ndarray(n,n) - distance matrix filled with zero for non-neighbors
    '''
    N = knn_inds.shape[0]
    R = np.zeros((N, N))
    for i in prange(N):
        R[i, knn_inds[i]] = distance[i]
    return R

def compDataD(knn_inds,R,N,tconorm="canonical",phi = None,phi_inv=None):
    r'''
    Symmetrizes a $(n,n)$ (sparse) matrix of nearest neighbor distances. The input matrix $R$ is assumed to contain
    non-zero entries only at nearest neighbor positions. We symmetrize by taking 
    a t-conorm T(a,b), where a is an element of R and b is an element or R.transpose()
    
    For the canonical t-conorm, this simplifies to the minimum whenever that is valid (if one of the distances is zero and does not correspond to a nearest neighbor, it is ignored)

    :param knn_inds: np.ndarray (n,k) - indices of nearest neighbors
    :param R: np.ndarray (n,n) - matrix with nearest neighbor distances filled by zeros elswhere
    :param tconorm: the t-conorm used for symmetrization - one of ['canonical', 'probabilistic sum', 'bounded sum', 'drastic sum', 'Einstein sum', 'nilpotent maximum']
    :return data_D: np.ndarray (n,n) - distance matrix
    '''

    if phi is None:
        phi = lambda x: torch.exp(-x)
        phi_inv = lambda x: -torch.log(x)
    if tconorm != "canonical":
        R = torch.from_numpy(R)
        R[R != 0] = phi(R[R != 0])
        RT = R.t()

        if tconorm == "probabilistic sum":
            data_D = R + RT - R * RT
        elif tconorm == "bounded sum":
            data_D = torch.minimum(R+RT,torch.ones_like(R))
        elif tconorm == "drastic sum":
            data_D = torch.zeros_like(R)
            data_D[(R != 0) & (RT == 0)] = R[(R != 0) & (RT == 0)]
            data_D[(R == 0) & (RT != 0)] = RT[(R == 0) & (RT != 0)]
            data_D[(R != 0) & (RT != 0)] = 1
        elif tconorm == "Einstein sum":
            data_D = (R+RT)/(1+R*RT)
        elif tconorm == "nilpotent maximum":
            data_D = torch.zeros_like(R)
            whereSumLowerOne = (R+RT<1)
            data_D[whereSumLowerOne] = torch.maximum(R[whereSumLowerOne],RT[whereSumLowerOne])
            data_D[~whereSumLowerOne] = torch.ones_like(R)[~whereSumLowerOne]
        else: # alternative implementation of canonical max-t-conorm
            data_D = torch.maximum(R,RT)
        # feel free to add your favourite t-conorm here

        data_D[data_D!=0] = phi_inv(data_D[data_D!=0])
        data_D = data_D.numpy()
        return data_D
    else: # the canonical (max) - t-conorm can be handled in a way that ensures that we spend at most Nxk operations:
        @njit(parallel=True) # use numba for parallelization
        def maxDataD(knn_inds,R,N):
            N = knn_inds.shape[0]
            data_D = np.zeros((N, N))

            for i in prange(N):
                for j in knn_inds[i]:
                    Rij = R[i, j]
                    RTij = R[j, i]

                    # we know that j is in the k-nns, so we check only if RTij is nonzero - if Rij is zero this means it is the nearest neighbor due to the normalization
                    if RTij != 0:
                        r = np.minimum(Rij, RTij)
                    else:
                        r = Rij
                    data_D[i, j] = r
                    data_D[j, i] = r
            return data_D
        return maxDataD(knn_inds,R,N)

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

def subMatrixEmbeddings(nc, SM, meanPointEmbeddings,**reduce_dim_kwargs):

    '''
    applys function reduce_dim to a list of submatrices, corresponding to connected components


    :param nc: nc: number of clusters/connected components
    :param SM: list of tuples (distance_matrices,indices)
    :param meanPointEmbeddings: embeddings of mean points of connected components
    :param reduce_dim_kwargs: arguments for reduce_dim function
    :return:
    '''
    subMatrixEmbeddings = []
    for i in range(nc):
        subMatrixEmbeddings.append(
            reduce_dim(SM[i][0],**reduce_dim_kwargs ))
        subMatrixEmbeddings[i] += meanPointEmbeddings[
            i]  # adds the meanPointEmbeddings[i]-vector to each row of the subMatrixEmbeddings[i]-matrix
    return subMatrixEmbeddings


def isumap(data,
           k: int,
           d: int,
           normalize:bool = True,
           distBeyondNN:bool = True,
           verbose: bool=True,
           dataIsDistMatrix: bool=False,
           dataIsGeodesicDistMatrix: bool = False,
           saveDistMatrix: bool = False,
           labels : bool=None,
           initialization="cMDS",
           metricMDS : bool =True,
           sgd_n_epochs:int= 1000,
           sgd_lr: float=1e-2,
           sgd_batch_size = None,
           sgd_max_epochs_no_improvement: int = 100,
           sgd_loss = 'MSE',
           sgd_saveplots_of_initializations:bool=True,
           sgd_saveloss: bool=False,
           tconorm = "canonical",
           phi = None,
           phi_inv = None,
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
    :param labels: cluster labels for plotting
    :param initialization: initialization for dimensionality reduction
    :param metricMDS: bool - if True, metricMDS is used for dimensionality reduction
    :param sgd_n_epochs: int - number of epochs for SGD, only used if metricMDS = True
    :param sgd_lr: float -lr for SGD, only used if metricMDS = True
    :param sgd_batch_size: int - batch size for SGD, only used if metricMDS=True
    :param sgd_max_epochs_no_improvement: int - number of epochs until early stopping, only used if metricMDS = True
    :param sgd_loss: loss for SGD (one of ['MSE','Sammon'] or custom loss)
    :param sgd_saveplots_of_initializations: bool - whether to save plots of SGD initialization
    :param sgd_saveloss:  bool - whether to save plots of SGD loss
    :param tconorm: the t-conorm used for symmetrization - one of ['canonical', 'probabilistic sum', 'bounded sum', 'drastic sum', 'Einstein sum', 'nilpotent maximum']
    :param phi: None or str or callable: phi function to transfer metrics to fuzzy weights. If None, defaults to exponential function.
    If a string, must be one of IMPLEMENTED_PHIS. Else, custom function may be used. Has to be callable, and an inverse phi_inv has to be provided.
    Does not do anything if t-conorm is 'canonical'.
    :param phi_inv: None or callable: inverse of phi function. If None, defaults to log. Else, must be callable inveres of phi.
    :param **phi_params**: additional parameters to pass to the phi function.
    :return finalEmbedding: np.ndarray (n,d) - points in low dimension
    :return clusterLabels: labels of connected components


    ...............................................

    Example Usage:

    X = np.random.normal((1000,10))
    Y = isumap(X,k=15,d=2)


    '''

    if (phi is not None) and (tconorm=='canonical'):
        warnings.warn('When using the canonical t-conorm, phi is irrelevant. If you intended to use a different phi, use one of the other t-conorms')



    if ((callable(phi) and (not callable(phi_inv)))) or ((callable(phi_inv) and (not callable(phi)))):

        raise TypeError('If you manually provide phi/phi_inv, you also have to provide the other one. Both have to be callable.')

    elif isinstance(phi,str):

        if phi not in IMPLEMENTED_PHIS:
            raise ValueError(
                f"Provided string does not match any of the currently implemented phis. "
                f"Valid choices are: {', '.join(IMPLEMENTED_PHIS)}"
            )
        if phi == 'exp':

            scale = phi_params.get('scale',1.0)
            phi = lambda x: torch.exp(-x/scale)
            phi_inv = lambda x: -torch.log(x)*scale


        elif phi =='half_normal':

            scale = phi_params.get('scale',1.0)
            sqrt2 = torch.sqrt(torch.tensor(2.0))
            phi = lambda x: 1.0-torch.erf(x/(sqrt2*scale))
            phi_inv = lambda x: sqrt2*scale*torch.erfinv(1.0-x)


        elif phi =='log_normal':
            scale = phi_params.get('scale', 1.0)
            sqrt2 = torch.sqrt(torch.tensor(2.0))

            phi = lambda x: 1.0- (1.0 + torch.erf(torch.log(x)/(sqrt2*scale)))/2.0
            phi_inv = lambda x: torch.exp(sqrt2*scale*torch.erfinv(1.0-2*x))

        elif phi=='pareto':

            scale = phi_params.get('scale',1.0)
            shape = phi_params.get('shape',2.0)

            phi = lambda x: torch.exp(-shape*torch.log1p(x / scale))
            phi_inv = lambda x: scale*(torch.exp(-torch.log1p(x-1.0) / shape)-1.0)
   
        elif phi =='uniform':

            if normalize == False:
                scale = data.max()

            else:
                scale = 1.0


            phi = lambda x: x/scale

            phi_inv = lambda x: scale*x




        
        
    if verbose:
        print("Number of CPU threads = ",cpu_count())
    N = data.shape[0]

    if dataIsGeodesicDistMatrix == False:
        t0 = time()
        if dataIsDistMatrix == True:
            print("Using precomputed distance matrix")
            knn_inds, knn_distances = find_nn_DistMatrix(data,k)
        else:
            knn_inds, knn_distances = find_nn(data,k)
        t1 = time()
        if verbose:
            printtime("Nearest neighbours computed in",t1-t0)

        t0 = time()
        distance = normalization(knn_distances,normalize,distBeyondNN)
        t1 = time()
        if verbose:
            printtime("Normalization computed in",t1-t0)

        t0=time()
        R = compR(knn_inds,distance)
        data_D = compDataD(knn_inds, R, N, tconorm = tconorm,phi=phi,phi_inv=phi_inv)
        t1 = time()
        if verbose:
            printtime("Neighbourhoods merged in",t1-t0)
        
        print("\nRunning Dijkstra...")
        t0 = time()
        graph = csr_matrix(data_D)
        partial_func = partial(dijkstra_wrapper, graph)
        D = []
        if __name__ == 'isumap':
            with Pool() as p:
                D = p.map(partial_func, range(N))
        D = np.array(D)
        t1 = time()
        if verbose:
            printtime("Dijkstra",t1-t0)
        
        if saveDistMatrix == True:
            if verbose:
                print("Storing geodesic distance matrix")
            with open('./Dataset_files/D.pkl', 'wb') as f:
                pickle.dump(D, f)
    else:
        if verbose:
            print("Using geodesic distance matrix")
        D = data

    t0 = time()
    SM = extractSubmatrices(D)
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
        meanPointEmbeddings = reduce_dim(clusterDistanceMatrix, d=d, n_epochs = sgd_n_epochs, lr=sgd_lr, batch_size = sgd_batch_size,max_epochs_no_improvement = sgd_max_epochs_no_improvement, loss = sgd_loss, initialization="cMDS", labels=labels, saveplots_of_initializations=False, metricMDS=False, saveloss=False, verbose=verbose, tconorm=tconorm)
        t1 = time()
        if verbose:
            printtime("Embedded the cluster mean points in",t1-t0)
    
    print("\nReducing dimension...")


    t0 = time()
    submatrixembeddings = subMatrixEmbeddings(nc, SM, meanPointEmbeddings,d=d, n_epochs=sgd_n_epochs, lr=sgd_lr, batch_size=sgd_batch_size,
                       max_epochs_no_improvement=sgd_max_epochs_no_improvement, loss=sgd_loss,
                       initialization=initialization, labels=labels,
                       saveplots_of_initializations=sgd_saveplots_of_initializations, metricMDS=metricMDS,
                       saveloss=sgd_saveloss, verbose=verbose, tconorm=tconorm)
    t1 = time()
    if verbose:
        printtime("Computed submatrix embeddings in",t1-t0)
    
    finalEmbedding = np.concatenate(submatrixembeddings)
    
    return  finalEmbedding, clusterLabels

