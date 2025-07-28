# see: https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html
# cells: bone marrow mononuclear cells of healthy human donors
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_CURRENT = os.path.join(SCRIPT_DIR, "../src/")
scriptPath = os.path.abspath(PATH_CURRENT)
sys.path.append(scriptPath)

from time import time
from isumap import isumap
from data_and_plots import plot_data, printtime, createNonUniformHemisphere, createMammoth, load_MNIST, createSwissRole, createFourGaussians, createMoons, createTorus, load_FashionMNIST
from multiprocessing import cpu_count
import numpy as np

from sklearn import datasets, manifold
N = 3000

k = 15
d = 2
# N = 2000
normalize = True
metricMDS = True
distBeyondNN = False
tconorm = "m_scheme"

if __name__ == '__main__':

    data, labels = datasets.make_s_curve(N, random_state=0)
    # data, labels = createNonUniformHemisphere(N)
    # data, labels = createSwissRole(N,hole=True,seed=0)
    # data, labels = createFourGaussians(8.2,N)
    # data, labels = createMoons(numberOfPoints,noise=0.1,seed=42)
    # data, labels = createTorus(N,seed=0)
    # data, labels = createMammoth(N,k=30,seed=42)

    # data, labels = load_MNIST(N)
    # data, labels = createBreastCancerDataset()
    # data, labels = load_FashionMNIST(N)

    # plot_data(data, labels, title="Initial dataset", display=True, save=False)

    t0=time()
    finalInitEmbedding, finalEmbedding, clusterLabels = isumap(data, k, d,
        normalize = normalize, distBeyondNN=distBeyondNN, verbose=True, dataIsDistMatrix=False, dataIsGeodesicDistMatrix = False, saveDistMatrix = False, initialization="cMDS", metricMDS=metricMDS, sgd_n_epochs = 1500, sgd_lr=1e-2, sgd_batch_size = None, sgd_max_epochs_no_improvement = 75, sgd_loss = 'MSE', sgd_saveloss=True, tconorm = tconorm, epm=True, m_scheme_value=1)
    t1 = time()
    printtime("Isumap total time",t1-t0)

    title = "Adata N_" + str(N) + " k_" + str(k) + " beyondNN_" + str(distBeyondNN) + " normalize_" + str(normalize) + " metricMDS_" + str(metricMDS) + " tconorm_" + tconorm

    plot_data(finalInitEmbedding,labels,title=title,display=True, save=False)
    plot_data(finalEmbedding,labels,title=title,display=True, save=False)
