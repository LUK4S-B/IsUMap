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

def spectral_embedding(M,d,verbose):
    if verbose:
        print("\nPerforming Spectral Embedding...")
    t0=time()
    spec_emb = SpectralEmbedding(n_components=d).fit_transform(M)
    t1 = time()
    if verbose:
        printtime("Spectral embedding",t1-t0)
    return spec_emb

def classical_multidimensional_scaling(M,d,verbose): 
    if verbose:
        print("\nPerforming classical MDS...")
    t0=time()
    @njit(parallel=True)
    def squareNormalize(M):
        M_squared = np.square(M,M)
        mean = np.sum(M_squared, axis=0) / M_squared.shape[0]
        return np.subtract(M_squared, mean)
    
    @njit(parallel=True)
    def compY(U,v,d):
        v[v<1e-14] = 0.0 # clean v in case of numerical errors
        L = np.diag(np.sqrt(v))
        Y = np.dot(U,L)/np.sqrt(d)
        return Y

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

def reduce_dim(D, d=2, n_epochs = 1000, lr=1e-2, batch_size = None,max_epochs_no_improvement = 100, loss = 'MSE', initialization="cMDS", labels=None, saveplots_of_initializations=True, metricMDS=True, saveloss=False, verbose=True, tconorm="canonical", clusternumber=0):
    if initialization=="cMDS":
        init = classical_multidimensional_scaling(D,d,verbose)
    elif initialization=="spectral":
        init = spectral_embedding(D,d,verbose)
    else:
        init = np.random.rand(D.shape[0],d)
        if verbose:
            print("Finished random initialization.")

    if saveplots_of_initializations:
        plot_data(init,labels,title = initialization + " initialization with N_" + str(D.shape[0]) + " tconorm_" + tconorm + " clusternum_" + str(clusternumber))
        print("Result of the initialization was stored in a file.\n")

    if metricMDS:
        print("\nPerforming metric MDS...")
        metric_mds_embedding = sgd_mds(D, init, n_epochs=n_epochs, lr=lr, batch_size=batch_size, max_epochs_no_improvement=max_epochs_no_improvement, loss=loss, saveloss=saveloss)
    else:
        metric_mds_embedding = init

    return metric_mds_embedding