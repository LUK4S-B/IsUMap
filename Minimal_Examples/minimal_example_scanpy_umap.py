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
import umap

k = 15
d = 2
N = 17041
normalize = True
metricMDS = True
distBeyondNN = False
tconorm = "canonical"

if __name__ == '__main__':

    adata = sc.read_h5ad("../../scanpy/adata_file_incl_leiden.h5ad")
    # adata = sc.read_h5ad("../../scanpy/s4d8_dimensionality_reduction_processed.h5ad")

    data = adata.obsm["X_pca"]
    labels = np.array(adata.obs["leiden"], dtype=int)

    print("data.shape: ", data.shape)
    print("labels.shape: ", labels.shape)
    print("")

    t0=time()
    finalEmbedding = umap.UMAP(random_state=42, n_neighbors=k).fit_transform(data)
    t1 = time()
    printtime("Isumap total time",t1-t0)
    
    title = "Adata UMAP file_incl_leiden N_" + str(N) + " k_" + str(k) + "leiden clusters"
    
    plot_data(finalEmbedding,labels,title=title,display=True, save=True)
    
    
    title = "Adata UMAP file_incl_leiden N_" + str(N) + " k_" + str(k) + "scDblFinder_class"
    labels = np.array(adata.obs["scDblFinder_class_labels"], dtype=int)

    plot_data(finalEmbedding,labels,title=title,display=True, save=True)
    
    
