import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors
from time import time
import pickle
from numba import njit, prange
from multiprocessing import Pool
from functools import partial
from dimensionReductionSchemes import reduce_dim
from data_and_plots import printtime

def dijkstra_wrapper(graph, i):
    return dijkstra(csgraph=graph, indices=i)

def isumap(data, k, d, normalize = True, distBeyondNN = True, verbose=True, dataIsDistMatrix=False, dataIsGeodesicDistMatrix = False, saveDistMatrix = False, labels=None, initialization="cMDS", metricMDS=True, sgd_n_epochs = 1000, sgd_lr=1e-2, sgd_batch_size = None,sgd_max_epochs_no_improvement = 100, sgd_loss = 'MSE', sgd_saveplots_of_initializations=True, sgd_saveloss=False):
    N = data.shape[0]
    
    @njit(parallel=True)
    def find_nn_DistMatrix(M, k):
        N = M.shape[0]
        knn_distances = np.empty((N, k))
        knn_inds = np.empty((N, k), dtype=np.int64)
        for i in prange(N):
            partitioned_indices = np.argpartition(M[i], k)
            knn_inds[i] = partitioned_indices[:k]
            knn_distances[i] = M[i][knn_inds[i]]
        return knn_inds, knn_distances

    def find_nn(data,k):
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(data)
        knn_distances, knn_inds = nn.kneighbors(data)
        return knn_inds, knn_distances

    @njit
    def two_smallest(numbers):
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
        return ind_m1,m2

    @njit(parallel=True)
    def normalization(distances,normalize,distBeyondNN):
        if distBeyondNN == True and normalize == True:
            for i in prange(distances.shape[0]):
                ind_m1,m2 = two_smallest(distances[i])
                distances[i] = distances[i] - m2
                distances[i][ind_m1] = 0.0
                mdi = np.max(distances[i])
                if 1e-10 < mdi:
                    distances[i] = distances[i] / mdi
        elif distBeyondNN == True and normalize == False:
            for i in prange(distances.shape[0]):
                ind_m1,m2 = two_smallest(distances[i])
                distances[i] = distances[i] - m2
                distances[i][ind_m1] = 0.0
        elif distBeyondNN == False and normalize == True:
            for i in prange(distances.shape[0]):
                mdi = np.max(distances[i])
                if 1e-10 < mdi:
                    distances[i] = distances[i] / mdi
        return distances
    
    @njit(parallel=True)
    def compR(knn_inds,distance,N):
        R = np.zeros((N,N))
        for i in prange(N):
            R[i,knn_inds[i]] = distance[i]
        return R
    
    @njit(parallel=True)
    def compDataD(knn_inds,R,N):
        RT = R.transpose()
        data_D = np.zeros((N,N))
        for i in prange(N):
            for j in knn_inds[i]:
                Rij = R[i,j]
                RTij = RT[i,j]
                if Rij != 0 and RTij != 0:
                    r = np.minimum(Rij,RTij)
                else:
                    r = np.maximum(Rij,RTij)
                data_D[i,j] = r
                data_D[j,i] = r
        return data_D
    
    @njit
    def extractSubmatrices(D):
        N = D.shape[0]
        S = np.array([i for i in range(N)])
        SMs = []
        i = 0
        while S.size != 0:
            indSet = np.where(D[S[0]] != np.inf)[0]
            n = len(indSet)
            SMs.append((np.empty((n,n)), indSet))
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
    def euclideanDistance(x,y):
        diff_xy = np.subtract(x,y)
        return np.sqrt(np.dot(diff_xy,diff_xy))

    @njit(parallel=True)
    def euclideanDistanceMatrixOfArray(pointArray):
        nc = pointArray.shape[0]
        distanceMatrix = np.zeros((nc,nc))
        for i in prange(nc):
            for j in range(i+1):
                if i==j:
                    distanceMatrix[i,j] = 0.0
                else:
                    distanceMatrix[i,j] = euclideanDistance(pointArray[i],pointArray[j])
                    distanceMatrix[j,i] = distanceMatrix[i,j]
        return distanceMatrix
    
    def compute_mean_points_and_labels(nc, data, SM):
        meanPointsOfClusters = np.empty((nc, data[0].size))
        clusterLabels = []
        for i in prange(nc):
            meanPointsOfClusters[i] = data[SM[i][1]].mean(0)
            clusterLabels.append(float(i)*np.ones(len(SM[i][1])))
        clusterLabels = np.concatenate(clusterLabels)
        return meanPointsOfClusters, clusterLabels
    
    def subMatrixEmbeddings(nc, SM, meanPointEmbeddings):
        subMatrixEmbeddings = []
        for i in range(nc):
            subMatrixEmbeddings.append(reduce_dim(SM[i][0], d=d, n_epochs = sgd_n_epochs, lr=sgd_lr, batch_size = sgd_batch_size,max_epochs_no_improvement = sgd_max_epochs_no_improvement, loss = sgd_loss, initialization=initialization, labels=labels, saveplots_of_initializations=sgd_saveplots_of_initializations, metricMDS=metricMDS, saveloss=sgd_saveloss,verbose=verbose))
            subMatrixEmbeddings[i] += meanPointEmbeddings[i] # adds the meanPointEmbeddings[i]-vector to each row of the subMatrixEmbeddings[i]-matrix
        return subMatrixEmbeddings

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
        R = compR(knn_inds,distance,N)
        data_D = compDataD(knn_inds, R, N)
        t1 = time()
        if verbose:
            printtime("Neighbourhoods merged in",t1-t0)

        print("Running Dijkstra...")
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
        meanPointEmbeddings = reduce_dim(clusterDistanceMatrix, d=d, n_epochs = sgd_n_epochs, lr=sgd_lr, batch_size = sgd_batch_size,max_epochs_no_improvement = sgd_max_epochs_no_improvement, loss = sgd_loss, initialization="cMDS", labels=labels, saveplots_of_initializations=False, metricMDS=False, saveloss=False, verbose=verbose)
        t1 = time()
        if verbose:
            printtime("Embedded the cluster mean points in",t1-t0)
    
    print("Reducing dimension...")
    t0 = time()
    subMatrixEmbeddings = subMatrixEmbeddings(nc, SM, meanPointEmbeddings)
    t1 = time()
    if verbose:
        printtime("Computed submatrix embeddings in",t1-t0)
    
    finalEmbedding = np.concatenate(subMatrixEmbeddings)
    
    return  finalEmbedding, clusterLabels

