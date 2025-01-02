from pylanczos import PyLanczos
from numba import njit
import numpy as np
from time import time
from sklearn.decomposition import PCA
from data_and_plots import plot_data, printtime
from metric_mds import sgd_mds
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.manifold import SpectralEmbedding

def spectral_embedding(M,d: int,verbose: bool):
    r'''

    :param M: np.ndarray(n,n) - distance matrix
    :param d: int - target dimensionality
    :param verbose: bool - prints info and timing if true
    :return: spec_emb: np.ndarray (n,d) - low-dimensional representation of datapoints
    '''
    if verbose:
        print("\nPerforming Spectral Embedding...")
    t0=time()
    spec_emb = SpectralEmbedding(n_components=d).fit_transform(M)
    t1 = time()
    if verbose:
        printtime("Spectral embedding",t1-t0)
    return spec_emb


@njit(parallel=True)
def squareNormalize(M):
    r'''Perform squaring and double centering,
    that is returns $CM^2C$, where $C=I-\frac{1}{n}\mathbb{1}_{n \times n}$

    :param M: np.ndarray(n,n) - distance matrix
    :return: M_squared np.ndarray(n,n) - squared and double centered distance matrix'''
    M_squared = np.square(M)
    N = M_squared.shape[0]
    M_squared = M_squared - np.sum(M_squared,axis=0).reshape((1,N))/N
    M_squared = M_squared - np.sum(M_squared,axis=1).reshape((N,1))/N
    return M_squared

@njit(parallel=True)
def compY(U, v, d):
    r'''
    Computes the classical multidimensional scaling (cMDS) embedding
    $Y = V^{\frac{1}{2}}U$

    :param U: np.ndarray(n,n)  - eigenvectors
    :param v: np.ndarray(n)    - eigenvalues
    :param d: int              - target dimension
    :return: Y: np.ndarray(n,d) - cMDS embedding
    '''
    v[v < 1e-14] = 0.0  # clean v in case of numerical errors 
    # and add a small number to prevent the points from collapsing to a single point in the embedding. Metric MDS can then push appart almost collapsed points.
    L = np.diag(np.sqrt(v))
    Y = np.dot(U, L) / np.sqrt(d)
    return Y


def classical_multidimensional_scaling(M,d: int,verbose:bool):
    r'''
        Performs classical multidimensional scaling on a distance matrix.
        This means that the for a given distance matrix $M$, we compute
        $$H = -\frac{1}{2} C (M^2) C$$
        where $C = I-\frac{1}{n}\mathbb{1}_{n\times n}$ is the centering matrix and the square is applied elementwise.
        A low-dimensional projection of the $d$ datapoints is then obtained by the first $d$ eigenvectors of $H$, multiplied
        with the square root of their eigenvalues.
        $Y = V^{\frac{1}{2}}U$
        where $V$ is a diagonal matrix of eigenvalues and $U$ is a matrix of eigenvectors


        :param M: np.ndarray(n,n) - distance matrix
        :param d: int - target dimensionality
        :param verbose: bool - prints info and timing if true
        :return: spec_emb: np.ndarray (n,d) - low-dimensional representation of datapoints
        '''
    if verbose:
        print("\nPerforming classical MDS...")
    t0=time()

    M = squareNormalize(M)

    if M.shape[0] < d: # handle the case where the number of points is lower than the embedding dimension
        v, U = PyLanczos(-M, True, M.shape[0]).run() 

        cMDS_Mshape = compY(U,v,M.shape[0])
        cMDS = np.zeros((M.shape[0],d))
        cMDS[:, :M.shape[0]] = cMDS_Mshape
    else:
        v, U = PyLanczos(-M, True, d).run() # Find d maximum eigenpairs. -M is the Gram matrix
        cMDS = compY(U,v,d)

    t1 = time()
    if verbose:
        printtime("classical MDS",t1-t0)
    return cMDS

def reduce_dim(D, d:int=2,
               n_epochs: int = 1000,
               lr: float=1e-2,
               batch_size = None,
               max_epochs_no_improvement: int = 100,
               loss = 'MSE',
               initialization='cMDS',
               metricMDS:bool=True,
               saveloss:bool=False,
               verbose:bool=True):
    r'''

    Applies a dimension reduction scheme to a distance matrix.
    Dimension reduction is either achieved via classical or metric multidimensional scaling (MDS), or via
    Laplacian eigenmaps (LE).
    The latter is implemented as a stochastic gradient descent here.


    :param D: np.ndarray (n,n) - distance matrix
    :param d: int - dimension of embedding
    :param n_epochs: int - number of epochs for SGD
    :param lr: float -lr for SGD
    :param batch_size: int - batch size for SGD
    :param max_epochs_no_improvement: int - number of epochs until early stopping for SGD
    :param loss: loss for SGD - one of ['MSE','Sammon'] or custom los
    :param initialization: initialization, must be one of ['cMDS','spectral','random'] or a np.ndarray.
                           if metricMDS =False, this is the final embedding that is used. 'spectral' is for LE.
    :param labels: cluster labels for connected components
    :param saveplots_of_initializations: bool - whether to save plots of data initialization
    :param metricMDS: bool - if True, metric MDS is performed
    :param saveloss: bool - if True, plots of SGD loss are saved
    :param verbose: bool - if True, prints progress and time info
    :param tconorm: t-conorm for merging - one of ['canonical', 'probabilistic sum', 'bounded sum', 'drastic sum', 'Einstein sum', 'nilpotent maximum']
    :return metric_mds_embedding: np.ndarray(n,d) - embedding of data points in d dimension.
    '''

    if initialization=='cMDS':
        init = classical_multidimensional_scaling(D,d,verbose)
    elif initialization=='spectral':
        init = spectral_embedding(D,d,verbose)
    elif initialization =='random':
        init = np.random.rand(D.shape[0],d)
        if verbose:
            print("Finished random initialization.")

    elif isinstance(initialization,np.ndarray):
        if initialization.shape == (D.shape[0], d):
            init = initialization
            initialization='custom'
        else:
            raise ValueError('Provided ndarray does not match required dimensions (D.shape[0], d).')
    else:
        raise ValueError(
            'Initialization must be one of "cMDS", "spectral", "random", or a NumPy array with shape (D.shape[0], d).')

    # add a bit of noise for stability:
    init += np.random.random(init.shape) * 1e-12

    if metricMDS:
        if verbose:
            print("\nPerforming metric MDS...")
        metric_mds_embedding = sgd_mds(D, init, n_epochs=n_epochs, lr=lr, batch_size=batch_size, max_epochs_no_improvement=max_epochs_no_improvement, loss=loss, saveloss=saveloss, verbose=verbose)
    else:
        metric_mds_embedding = init

    return init, metric_mds_embedding