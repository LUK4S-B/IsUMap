import sys
import os
SCRIPT_DIR = os.getcwd()
PATH_CURRENT = os.path.join(SCRIPT_DIR, "../../src/")
PATH_CURRENT2 = os.path.join(SCRIPT_DIR, "../../src/cluster_mds/")
PLOT_PATH = os.path.join(SCRIPT_DIR, "../results/Simulations for article - Fuzzy simplicial sets and their application to geometric data analysis/")
scriptPath = os.path.abspath(PATH_CURRENT)
scriptPath2 = os.path.abspath(PATH_CURRENT2)
sys.path.append(scriptPath)
sys.path.append(scriptPath2)
datasetPath = os.path.join(SCRIPT_DIR, "../../Dataset_files/")

from metrics import *
from isumap import isumap
from isumap_cluster import isumap_cluster
from clusterSeparationOptimizer import visualize_optimization, visualize_optimization_with_labels
from data_and_plots import plot_data, printtime, load_MNIST, load_FashionMNIST, createMammoth, createNonUniformHemisphere, createSwissRoll, createTorus, make_s_curve_with_hole, createSCurve
from time import time
import numpy as np
import pickle

# Storing
def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

# Loading
def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def run_isumap_cluster(data, labels, k=15,distBeyondNN=False,normalize=True,tconorm="canonical",epm=True,distFun="canonical", datasetName="", save_results=True, dataIsGeodesicDistMatrix=False, dataIsDistMatrix=False, saveDistMatrix=False,cluster_algo="linkage_cluster",**kwargs):
    N = data.shape[0]

    t0=time()
    results = isumap_cluster(data, k,
        normalize = normalize, distBeyondNN=distBeyondNN, tconorm = tconorm, epm=epm, distFun=distFun,verbose=True, also_return_optimizer_model=True, also_return_medoid_paths=True, labels=labels, dataIsGeodesicDistMatrix=dataIsGeodesicDistMatrix, dataIsDistMatrix=dataIsDistMatrix, saveDistMatrix=saveDistMatrix,cluster_algo=cluster_algo,**kwargs)
    t1 = time()
    printtime("Isumap total time",t1-t0)

    optimizer_model = results['optimizer_model']
    medoid_paths = results['medoid_paths']
    reordered_labels_split_into_clusters = results['true_labels_split_into_clusters']

    title = datasetName + " N_" + str(N) + " k_" + str(k) + " beyondNN_" + str(distBeyondNN) + " normalize_" + str(normalize) + " tconorm_" + tconorm + " distFun_" + distFun + " epm_" + str(epm)

    if save_results:
        my_object = {"optimizer_model": optimizer_model, "medoid_paths": medoid_paths, "title": title, "labels": labels, "reordered_labels_split_into_clusters": reordered_labels_split_into_clusters} 
        save_object(my_object, datasetName + '_projection.pkl')

    return optimizer_model, medoid_paths, title, labels, reordered_labels_split_into_clusters

def run_isumap(data, labels, k=15, distBeyondNN=False,normalize=True,tconorm="canonical",epm=True,distFun="canonical", datasetName="", save_results=True, dataIsGeodesicDistMatrix=False, dataIsDistMatrix=False, saveDistMatrix=False,cluster_algo="linkage_cluster",plot_path=PLOT_PATH,plotMetricMDS=False,**kwargs):
    N = data.shape[0]

    t0=time()
    finalInitEmbedding, finalEmbedding, clusterLabels = isumap(data, k, 2,
        normalize = normalize, distBeyondNN=distBeyondNN, tconorm = tconorm, epm=epm, distFun=distFun,verbose=True, also_return_optimizer_model=True, also_return_medoid_paths=True, labels=labels, dataIsGeodesicDistMatrix=dataIsGeodesicDistMatrix, dataIsDistMatrix=dataIsDistMatrix, saveDistMatrix=saveDistMatrix,cluster_algo=cluster_algo,**kwargs)
    t1 = time()
    printtime("Isumap total time",t1-t0)

    title = datasetName + " N_" + str(N) + " k_" + str(k) + " beyondNN_" + str(distBeyondNN) + " normalize_" + str(normalize) + " tconorm_" + tconorm + " distFun_" + distFun + " epm_" + str(epm)

    if plotMetricMDS:
        plot_data(finalEmbedding, labels, title=title, display=True, save=save_results, path=plot_path)

        return finalEmbedding
    else:
        plot_data(finalInitEmbedding, labels, title=title, display=True, save=save_results, path=plot_path)

        return finalInitEmbedding

def plot_isumap_cluster(optimizer_model, medoid_paths, title, plot_path=PLOT_PATH, enable_grid=True, labels=None, custom_color_map='jet'):

    visualize_optimization(optimizer_model, title=title, medoid_paths=medoid_paths, save_path=plot_path+"/"+title + " grid_" + str(enable_grid)+".png", display=True, enable_grid=enable_grid)
    visualize_optimization_with_labels(optimizer_model, title=title, medoid_paths=medoid_paths, save_path=plot_path+"/"+title+"_with_cluster_labels grid_" + str(enable_grid)+".png", display=True, enable_grid=enable_grid, custom_color_map=custom_color_map)
    if labels is not None:
        visualize_optimization_with_labels(optimizer_model, title=title, medoid_paths=medoid_paths, save_path=plot_path+"/"+title+"_with_labels grid_" + str(enable_grid)+".png", display=True, enable_grid=enable_grid, labels=labels, custom_color_map=custom_color_map)

def load_my_object(datasetName):
    loaded_object = load_object(datasetName + "_projection.pkl")
    optimizer_model = loaded_object["optimizer_model"]
    medoid_paths = loaded_object["medoid_paths"]
    title = loaded_object["title"]
    labels = loaded_object["labels"]
    reordered_labels_split_into_clusters = loaded_object["reordered_labels_split_into_clusters"]
    return optimizer_model, medoid_paths, title, labels, reordered_labels_split_into_clusters