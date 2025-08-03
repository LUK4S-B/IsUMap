import os
import sys
import numpy as np

# Set the path to the directory containing `isumap.py` 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_CURRENT = os.path.join(SCRIPT_DIR, "../src/")
scriptPath = os.path.abspath(PATH_CURRENT)
sys.path.append(scriptPath)

from isumap_cluster import isumap_cluster
from data_and_plots import plot_data, createMammoth, load_MNIST, printtime, createNonUniformHemisphere, createSwissRole, createFourGaussians, createMoons, createTorus, load_FashionMNIST, createBreastCancerDataset, createSCurve

from multiprocessing import cpu_count
from time import time

k = 15
N = 3000

epm = True
normalize = True
distBeyondNN = False
tconorm = "canonical"
distFun = "canonical"


if __name__ == '__main__':
    title = "MNIST 3D N_" + str(N) + " k_" + str(k) + " beyondNN_" + str(distBeyondNN) + " normalize_" + str(normalize) + " tconorm_" + tconorm + " distFun_" + distFun + " phi_exp" + " epm_" + str(epm)

    # data, labels = createNonUniformHemisphere(N)
    # data, labels = createSwissRole(N,hole=True,seed=0)
    # data, labels = createFourGaussians(8.2,N)
    # data, labels = createMoons(N,noise=0.1,seed=42)
    # data, labels = createTorus(N,seed=0)
    # data, labels = createMammoth(N,k=30,seed=42)
    data, labels = load_MNIST(N)
    # data, labels = createBreastCancerDataset()
    # data, labels = load_FashionMNIST(N)
    # data, labels = createSCurve(N, noise=0.1, seed=42)
    # data, labels = load_CIFAR_10(N)

    # plot_data(data,labels,title="Initial dataset",display=True, save=False)
    
    t0=time()
    results = isumap_cluster(data, k,
        normalize = normalize, distBeyondNN=distBeyondNN, tconorm = tconorm, distFun=distFun, epm=epm, cluster_algo = "linkage_cluster", labels = labels, preprocess_with_pca = False, pca_components = 40, plot_original_data = False)
    t1 = time()
    
    # plot_data(finalInitEmbedding,labels,title=title+" init",display=True, save=True)
    # plot_data(finalEmbedding,labels,title=title,display=True, save=True)
    # print("\nResult saved in './Results/" + title + ".png'")
    
    printtime("Isumap total time",t1-t0)
    
