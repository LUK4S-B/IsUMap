from time import time
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.sparse import csr_matrix, lil_matrix, save_npz
from scipy.sparse.csgraph import dijkstra
import scipy.special as s
from sklearn.neighbors import NearestNeighbors
import pickle
import warnings
from data_and_plots import printtime
from numba import njit, prange

IMPLEMENTED_PHIS = ['exp','half_normal','log_normal','pareto','uniform','identity']

def dijkstra_wrapper(graph, directedDistances, i):
    return dijkstra(csgraph=graph, indices=i, directed=directedDistances, return_predecessors=False)

@njit(parallel=True)
def find_nn_DistMatrix(M, k:int):
    r'''
    Implements an exact k nearest neighbour search.

    :param M: np.ndarray(n,n) - distance matrix
    :param k: int - number of nearest neighbors
    :return knn_inds: np.ndarray(n,k) - indices of nearest neighbors
    :return knn_distances: np.ndarray(n,d) - distances of nearest neighbors

    '''
    # N = M.shape[0]
    # knn_distances = np.empty((N, k))
    # knn_inds = np.empty((N, k), dtype=np.int64)
    # for i in prange(N):
    #     partitioned_indices = np.argpartition(M[i], k)
    #     knn_inds[i] = partitioned_indices[:k]
    #     knn_distances[i] = M[i][knn_inds[i]]
    # return knn_inds, knn_distances
    N = M.shape[0]
    knn_distances = np.empty((N, k))
    knn_inds = np.empty((N, k), dtype=np.int64)
    for i in prange(N):
        partitioned_indices = np.argpartition(M[i], k)
        k_nearest_indices = partitioned_indices[:k]
        k_nearest_distances = M[i][k_nearest_indices]
        
        # Sort the k nearest by distance
        sort_order = np.argsort(k_nearest_distances)
        knn_inds[i] = k_nearest_indices[sort_order]
        knn_distances[i] = k_nearest_distances[sort_order]
        
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
            distances[i] = distances[i] - m2
            distances[i][ind_m1] = 0.0
    elif distBeyondNN == False and normalize == True:
        for i in prange(distances.shape[0]):
            mdi = np.max(distances[i])
            if 1e-10 < mdi:
                distances[i] = distances[i] / mdi
    return distances

@njit
def euclidean_dist(knn_distances, knn_inds, data, i, j, k, ind_j, ind_k):
    p0 = data[i]
    p1 = data[j]
    p2 = data[k]

    v1 = p1-p0
    v2 = p2-p0
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    p1n = p0 + v1 * knn_distances[i][ind_j]
    p2n = p0 + v2 * knn_distances[i][ind_k]

    d = np.sqrt(np.sum(np.square(p1n-p2n)))
    return d

@njit
def canonical_dist(knn_distances, knn_inds, data, i, j, k, ind_j, ind_k):
    return knn_distances[i][ind_j]+knn_distances[i][ind_k]

@njit
def freedman_dist(knn_distances, knn_inds, data, i, j, k, ind_j, ind_k):
    d = (knn_distances[i][ind_j]+knn_distances[i][ind_k])/np.sqrt(2)
    return d

def comp_graph(knn_inds, knn_distances, data, f, epm, directedDistances):
    N = knn_inds.shape[0]
    R = {} # R is a dictionary and R[(i,j,k)] contains the distance from point j to k in neighborhood i. Non-symmetrized. R is sparse in the sense that it does not contain distances that are infinite or when j==k
    for i in range(N):
        for ind_j,j in enumerate(knn_inds[i]):
            for ind_k,k in enumerate(knn_inds[i]):
                if directedDistances:
                    if j==i:
                        R[(i,j,k)] = knn_distances[i][ind_k]
                    else:
                        if not epm: # epm == True leads to pure star graphs
                            R[(i,j,k)] = f(knn_distances, knn_inds, data, i, j, k, ind_j, ind_k)
                else:
                    if j<k:
                        if j==i:
                            R[(i,j,k)] = knn_distances[i][ind_k]
                        elif k==i:
                            R[(i,j,k)] = knn_distances[i][ind_j]
                        else:
                            if not epm: # epm == True leads to pure star graphs
                                R[(i,j,k)] = f(knn_distances, knn_inds, data, i, j, k, ind_j, ind_k)
    return R

def apply_t_conorm_recursively(graph, tconorm, N, phi, phi_inv, m_scheme_value = 1.0, save_fuzzy_graph = False):
    m_scheme_value = phi(m_scheme_value)
    if tconorm == "probabilistic sum":
        def T_conorm(a,b):
            return a+b-a*b
            # if (a == 1.0) or (b == 1.0):
            #     return 1.0
            # else:
            #     return 1.0 - np.exp(np.log(1.0-a) + np.log(1.0-b)) # equals a+b-ab but is more numerically stable when ab is very small.
    elif tconorm == "bounded sum":
        def T_conorm(a,b):
            return min(a+b,1)
    elif tconorm == "drastic sum":
        def T_conorm(a,b):
            if a==0:
                return b
            elif b==0:
                return a
            else:
                return 1
    elif tconorm == "Einstein sum":
        def T_conorm(a,b):
            return (a+b)/(1+a*b)
    elif tconorm == "nilpotent maximum":
        def T_conorm(a,b):
            if a+b < 1:
                return max(a,b)
            else:
                return 1

    # Note that this reduces to the canonical t-conorm if m_scheme_value is not specified
    elif tconorm == "m_scheme":
        max_distance_in_graph = max(graph.values())
        def T_conorm(a,b): 
            if a == np.inf:
                return b
            elif b == np.inf:
                return a
            else:
                return min(a,b,m_scheme_value*max_distance_in_graph)
    elif tconorm == "m_scheme_Wiener_Shannon":
        def T_conorm(a,b): 
            return max(- np.log(np.exp(-m_scheme_value*a) + np.exp(-m_scheme_value*b)) / m_scheme_value , 0)
    elif tconorm == "m_scheme_Composition":
        def T_conorm(a,b): 
            return max(- np.log(np.exp(-m_scheme_value*a) + np.exp(-m_scheme_value*b) - np.exp(-m_scheme_value*(a+b))) / m_scheme_value, 0) # here max(..., 0) is inserted for numerical stability. Finite numerical precision can lead to np.exp(-m_scheme_value*a) + np.exp(-m_scheme_value*b) - np.exp(-m_scheme_value*(a+b)) bigger 1.
    elif tconorm == "m_scheme_Hyperbolic":
        def T_conorm(a,b):
            if a == np.inf and b == np.inf:
                return np.inf
            elif a == np.inf or b == np.inf:
                return 1.0
            elif a == 0 and b == 0:
                return 0.0
            else:
                return a*b / (a+b)

    # feel free to add your favourite t-conorm here

    else: # canonical max-t-conorm
        def T_conorm(a,b):
            return max(a,b)

    g = lil_matrix((N,N))

    for key, value in graph.items():
        if tconorm.startswith("m_scheme"):
            if g[key[1],key[2]] == 0:
                g[key[1],key[2]] = np.inf  # this is necessary because we use sparse matrices that can not be initialized with inf-values. However, all values that remain entirely 0 are treated as if they were inf by Dijkstra later. Hence, this operation only has to be performed for the case in which some-non-infinite value exists at that key in the graph dictionary
        g[key[1],key[2]] = T_conorm(phi(value),g[key[1],key[2]]) + np.nextafter(0., np.float32(1.))
        if np.isnan(g[key[1],key[2]]):
            raise ValueError("Error: g[key[1],key[2]] is NaN. Perhaps division by 0 or inf occured in some t-conorm or m-scheme.")

    if save_fuzzy_graph:
        save_npz("graph_before_inv.npz", g.tocsr())

    for i, (row_indices, row_data) in enumerate(zip(g.rows, g.data)): 
        for j, value in zip(row_indices, row_data):
            g[i, j] = phi_inv(value) + np.nextafter(0., np.float32(1.)) # np.nextafter(0., np.float32(1.)) adds the smallest possible floating point number, which is necessary for scipy's Dijkstra-routine, which treats exact 0's like Infinity (because it uses sparse matrices)

    return g.tocsr()

def distance_graph_generation(data,
           k: int,
           normalize:bool = True,
           distBeyondNN:bool = True,
           verbose: bool = True,
           dataIsDistMatrix: bool = False,
           directedDistances: bool = False,
           dataIsGeodesicDistMatrix: bool = False,
           saveDistMatrix: bool = False,
           tconorm = "canonical",
           distFun = "canonical",
           phi = None,
           phi_inv = None,
           epm = True,
           m_scheme_value = 1.0,
           save_fuzzy_graph = False,
           apply_Dijkstra = True,
           max_param = 100.0,
           **phi_params):
    '''

    distance_graph_generation generates an n x n distance matrix, where n is the number of provided data points. The distances are generated with the following steps:
        1. Local neighborhoods are extracted from the datapoints,
        2. The distances in those are normalized if desired
        3. Local neighborhoods are merged, for example using T-conorms
        4. The Dijkstra algorithm is used if desired to compute the geodesics of the merged graph

    :param data: np.ndarray (n,m) - matrix of data points
    :param k: int - number of nearest neighbors to use in distance computation
    :param normalize: bool - if True, data is normalized by k-nn distance
    :param distBeyondNN: bool - if True, distance of nearest neighbor is subtracted from distances
    :param verbose: bool - if True, prints progress and time information
    :param dataIsDistMatrix: bool - whether data is a distance matrix, if False, distance is computed from data
    :param dataIsGeodesicDistMatrix: bool - whether data is geodesic distance matrix, if False, dijkstra is applied
    :param saveDistMatrix:  bool -whether to save distance matrix
    :param tconorm: the t-conorm used for symmetrization - one of ['canonical', 'probabilistic sum', 'bounded sum', 'drastic sum', 'Einstein sum', 'nilpotent maximum']
    :param phi: None or str or callable: phi function to transfer metrics to fuzzy weights. If None, defaults to exponential function.
    If a string, must be one of ['exp','half_normal','log_normal','pareto','uniform']. Else, custom function may be used. Has to be callable, and an inverse phi_inv has to be provided.
    Does not do anything if t-conorm is 'canonical'.
    :param phi_inv: None or callable: inverse of phi function. If None, defaults to log. Else, must be callable inverse of phi.
    :param epm: If this parameter is set to True, then all non-radial distances in the star graph are set to Infinity as if we were in the category EPMet. Default is False.
    :param save_fuzzy_graph: Safes the fuzzy graph before conversion to a metric space if desired.
    :param apply_Dijkstra: If set to False, then no geodesic graph-hopping distances are added to the final distance matrix.
    :param **phi_params**: additional parameters to pass to the phi function.
    :return D: np.ndarray (n,n) - n x n distance graph. Can be directed if desired.

    ...............................................

    Example Usage:

    X = np.random.normal((1000,10))
    D = distance_graph_generation(X,k=15)


    '''
    
    if tconorm.startswith("m_scheme"):
        if phi != None:
            warnings.warn("When using m_schemes, phi must equal the identity. The specified phi is not going to have any effect. Use other tconorms instead if you want to make use of phi.")
        phi = 'identity'

    if m_scheme_value <= 0:
        raise ValueError("Error: m_scheme_value should be > 0.")

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
            phi = lambda x: np.exp(-x/scale)
            phi_inv = lambda x: -np.log(x)*scale

        elif phi =='half_normal':
            scale = phi_params.get('scale',1.0)
            sqrt2 = np.sqrt(2.0)
            phi = lambda x: 1.0-s.erf(x/(sqrt2*scale))
            phi_inv = lambda x: sqrt2*scale*s.erfinv(1.0-x)

        elif phi =='log_normal':
            scale = phi_params.get('scale', 1.0)
            sqrt2 = np.sqrt(2.0)
            phi = lambda x: 1.0- (1.0 + s.erf(np.log(x)/(sqrt2*scale)))/2.0
            phi_inv = lambda x: np.exp(sqrt2*scale*s.erfinv(1.0-2*x))

        elif phi == 'pareto':
            scale = phi_params.get('scale',1.0)
            shape = phi_params.get('shape',2.0)
            phi = lambda x: np.exp(-shape*s.log1p(x / scale))
            phi_inv = lambda x: scale*(np.exp(-s.log1p(x-1.0) / shape)-1.0)

        elif phi == 'uniform':
            if normalize == False:
                scale = data.max()
            else:
                scale = 1.0
            phi = lambda x: 1.0 - min(x/scale,1.0)
            phi_inv = lambda x: (1.0-min(x,1.0))*scale

        elif phi == 'identity':
            phi = lambda x: x
            phi_inv = phi

    elif phi is None:
        scale = phi_params.get('scale',1.0)
        phi = lambda x: np.exp(-x/scale)
        phi_inv = lambda x: -np.log(x)*scale

    if verbose:
        print("Number of CPU threads = ",cpu_count())
    N = data.shape[0]

    if not dataIsGeodesicDistMatrix:
        t0 = time()
        if dataIsDistMatrix:
            print("Using precomputed distance matrix")
            knn_inds, knn_distances = find_nn_DistMatrix(data,k)
        else:
            knn_inds, knn_distances = find_nn(data,k)
        t1 = time()
        if verbose:
            printtime("Nearest neighbours computed in",t1-t0)

        t0 = time()
        knn_distances = normalization(knn_distances,normalize,distBeyondNN)
        t1 = time()
        if verbose:
            printtime("Normalization computed in",t1-t0)

        t0=time()
        if verbose:
            print("Computing the graph...")
        if distFun == "freed":
            neighborhood_dist_function = freedman_dist
        elif distFun == "euclidean":
            neighborhood_dist_function = euclidean_dist
        else:
            neighborhood_dist_function = canonical_dist
        data_D = comp_graph(knn_inds, knn_distances, data, neighborhood_dist_function, epm, directedDistances)
        t1=time()
        if verbose:
            printtime("Graph computation",t1-t0)

        t0 = time()
        if verbose:
            print("Applying t-conorm...")
        graph = apply_t_conorm_recursively(data_D,tconorm,N, phi, phi_inv, m_scheme_value, save_fuzzy_graph)
        t1=time()
        if verbose:
            printtime("T-conorm application",t1-t0)
        
        # if saveDistMatrix == True:
        #     if verbose:
        #         print("Storing distance matrix before geodesics")
        #     with open('./Dataset_files/D_before_geod.pkl', 'wb') as f:
        #         pickle.dump(graph.todense(), f)

        if apply_Dijkstra:
            if verbose:
                print("\nRunning Dijkstra...")
            t0 = time()
            partial_func = partial(dijkstra_wrapper, graph, directedDistances)
            D = []
            if __name__ == 'distance_graph_generation':
                with Pool() as p:
                    D = p.map(partial_func, range(N))
                    p.close()
                    p.join()
            D = np.array(D)
            t1 = time()
            if verbose:
                printtime("Dijkstra",t1-t0)
        else:
            D = graph.toarray()
            diam = np.max(D)
            D[D==0] = max(diam,max_param)
            np.fill_diagonal(D, 0)
        
        if saveDistMatrix == True:
            if verbose:
                print("Storing geodesic distance matrix")
            with open('./Dataset_files/D.pkl', 'wb') as f:
                pickle.dump(D, f)
    else:
        if verbose:
            print("Using geodesic distance matrix")
        D = data

    return D