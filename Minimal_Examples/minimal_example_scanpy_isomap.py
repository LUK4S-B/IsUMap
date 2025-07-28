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
from data_and_plots import plot_data, printtime, createNonUniformHemisphere #, createMammoth, load_MNIST, createSwissRole, createFourGaussians, createMoons, createTorus, load_FashionMNIST
from multiprocessing import cpu_count
import scanpy as sc
import numpy as np

k = 15
d = 2
N = 17041
normalize = False
metricMDS = True
distBeyondNN = False
tconorm = "canonical"

if __name__ == '__main__':

    # adata = sc.read_h5ad("../../scanpy/adata_file_incl_leiden.h5ad")
    adata = sc.read_h5ad("../../scanpy/s4d8_dimensionality_reduction_processed.h5ad")

    data = adata.obsm["X_pca"]
    labels = np.array(adata.obs["leiden"], dtype=int)

    print("data.shape: ", data.shape)
    print("labels.shape: ", labels.shape)
    print("")

    t0=time()
    finalInitEmbedding, finalEmbedding, clusterLabels = isumap(data, k, d,
        normalize = normalize, distBeyondNN=distBeyondNN, verbose=True, dataIsDistMatrix=False, dataIsGeodesicDistMatrix = False, saveDistMatrix = False, initialization="cMDS", metricMDS=metricMDS, sgd_n_epochs = 1500, sgd_lr=1e-2, sgd_batch_size = None, sgd_max_epochs_no_improvement = 75, sgd_loss = 'MSE', sgd_saveloss=True, tconorm = tconorm, epm=True)
    t1 = time()
    printtime("Isumap total time",t1-t0)
    
    title = "Adata isomap s4d8_dimensionality_reduction N_" + str(N) + " k_" + str(k) + " beyondNN_" + str(distBeyondNN) + " normalize_" + str(normalize) + " metricMDS_" + str(metricMDS) + " tconorm_" + tconorm + "leiden clusters"
    
    plot_data(finalInitEmbedding,labels,title=title+" init",display=True, save=True)
    plot_data(finalEmbedding,labels,title=title,display=True, save=True)
    
    
    title = "Adata isomap s4d8_dimensionality_reduction N_" + str(N) + " k_" + str(k) + " beyondNN_" + str(distBeyondNN) + " normalize_" + str(normalize) + " metricMDS_" + str(metricMDS) + " tconorm_" + tconorm + "scDblFinder_class"
    labels = np.array(adata.obs["scDblFinder_class_labels"], dtype=int)

    plot_data(finalInitEmbedding,labels,title=title+" init",display=True, save=True)
    plot_data(finalEmbedding,labels,title=title,display=True, save=True)
    
    
