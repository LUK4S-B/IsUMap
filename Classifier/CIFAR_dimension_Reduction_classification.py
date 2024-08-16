from Module_classifier import compute_classification_acc
import pickle

from time import time
import os
import sys

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import random
import tensorflow_datasets as tfds

def load_cifar10_data(pickle_file='./../Dataset_files/cifar10_data.pkl', N=100):
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
    else:
        data = tfds.load('cifar10', split=['train', 'test'], as_supervised=True)
        data = list(data[0]) + list(data[1])
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)

    # Convert data to numpy arrays
    x_data = np.array([np.array(image) / 255.0 for image, label in data])
    y_data = np.array([label.numpy() for image, label in data])

    # Take a random subset of N images and labels
    indices = random.sample(range(len(x_data)), N)
    data_subset = x_data[indices]
    labels_subset = y_data[indices]

    data_subset = data_subset.reshape(N, -1)

    return data_subset, labels_subset

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../")
from data_and_plots import printtime
from isumap import isumap

def isu(data, labels, k, d, verbose=False, normalize = False, metricMDS = True, distBeyondNN = False, tconorm = "canonical"):
    if __name__ == '__main__':
        t0=time()
        emb, clusterLabels = isumap(data, k, d,
            normalize = normalize, distBeyondNN=distBeyondNN, verbose=verbose, dataIsDistMatrix=False, dataIsGeodesicDistMatrix = False, saveDistMatrix = False, labels=labels, initialization="cMDS", metricMDS=metricMDS, sgd_n_epochs = 1500, sgd_lr=1e-2, sgd_batch_size = None,sgd_max_epochs_no_improvement = 75, sgd_loss = "MSE", sgd_saveplots_of_initializations=False, sgd_saveloss=False, tconorm = tconorm)
        t1 = time()
        total_time = t1-t0
        printtime("IsUMap total time",total_time)
    return emb, total_time

def iso(data, labels, k, d, verbose=False):
    if __name__ == '__main__':
        t0=time()
        emb, clusterLabels = isumap(data, k, d,
            normalize = False, distBeyondNN=False, verbose=verbose, dataIsDistMatrix=False, dataIsGeodesicDistMatrix = False, saveDistMatrix = False, labels=labels, initialization="cMDS", metricMDS=False, sgd_n_epochs = 1500, sgd_lr=1e-2, sgd_batch_size = None, sgd_max_epochs_no_improvement = 75, sgd_loss = "MSE", sgd_saveplots_of_initializations=False, sgd_saveloss=False, tconorm = "canonical")
        t1 = time()
        total_time = t1-t0
        printtime("Isomap total time",total_time)
    return emb, total_time

def uma(data,k,d,verbose=False):
    reducer = umap.UMAP(n_components=d,n_neighbors=k, verbose=verbose)
    t0=time()
    emb = reducer.fit_transform(data)
    t1 = time()
    total_time = t1-t0
    printtime("UMAP total time",total_time)

    return emb, total_time

def pc(data,d,verbose=False):
    if verbose:
        print("Performing PCA")
    t0=time()
    emb = PCA(n_components=d).fit_transform(data)
    t1 = time()
    total_time = t1-t0
    printtime("PCA total time",total_time)
    return emb, total_time

def tsn(data,d,verbose=False):
    if verbose:
        print("Performing t-SNE")
    t0=time()
    emb = TSNE(n_components=d, random_state=42, perplexity=40).fit_transform(data)
    t1 = time()
    total_time = t1-t0
    printtime("t-SNE total time",total_time)
    return emb, total_time


def compute_acc_for_method(data, labels, k, d,  method, linerClassifier, N, output_dim, verbose=False, normalize = False, metricMDS = True, distBeyondNN = False, tconorm = "canonical"):
    if method=='isumap':
        emb, total_emb_time = isu(data, labels, k, d, verbose=verbose, normalize = normalize, metricMDS = metricMDS, distBeyondNN = distBeyondNN, tconorm = tconorm)
    elif method=='umap':
        emb, total_emb_time = uma(data, k, d, verbose=verbose)
    elif method=='isomap':
        emb, total_emb_time = iso(data, labels, k, d, verbose=verbose)
    elif method=='pca':
        emb, total_emb_time = pc(data,d,verbose=verbose)
    elif method=='tsne':
        emb, total_emb_time = tsn(data,d,verbose=verbose)

    test_emb = emb[N:2*N]
    test_labels = labels[N:2*N]
    train_emb = emb[0:N]
    train_labels = labels[0:N]

    t0 = time()
    accuracy, test_accuracy = compute_classification_acc(train_emb,train_labels,test_emb,test_labels,linerClassifier,output_dim=output_dim, verbose=verbose)
    t1 = time()
    total_classification_time = t1-t0
    printtime("Classification time",total_classification_time)

    return accuracy, test_accuracy, total_emb_time, total_classification_time


def compute_results(ds,method,N,k,linerClassifier,output_dim,verbose=False, normalize = False, metricMDS = True, distBeyondNN = False, tconorm = "canonical"):
    data, labels = load_cifar10_data(N=2*N)
    train_accs = []
    test_accs = []
    dim_red_times = []
    classification_times = []
    for d in ds:
        train_acc, test_acc, total_emb_time, total_class_time = compute_acc_for_method(data, labels, k, d,  method, linerClassifier, N, output_dim, verbose=verbose, normalize = normalize, metricMDS = metricMDS, distBeyondNN = distBeyondNN, tconorm = tconorm)
        if verbose:
            print("\n\n\nAccuracies: ",train_acc, test_acc)
            print("Times: ", total_emb_time, total_class_time)
            print(method,"d =",d)
            print("\n")
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        dim_red_times.append(total_emb_time)
        classification_times.append(total_class_time)
    results = {'ds': ds, 'method':method, 'N':N, 'k': k, 'train_accs': train_accs, 'test_accs': test_accs, 'dim_red_times': dim_red_times, 'classification_times': classification_times, 'dataset': 'CIFAR', 'linerClassifier': linerClassifier, 'output_dim': output_dim, 'verbose': verbose, 'normalize' : normalize, 'metricMDS': metricMDS, 'distBeyondNN': distBeyondNN, 'tconorm': tconorm}
    return results

ds = [2,3,5,10,20,50,100,200,400]
linerClassifier = True
N = 4000
k = 15
output_dim = 10

methods = ['isumap','umap','isomap']

rs = {}
for method in methods:
    r = compute_results(ds,method,N,k,linerClassifier,output_dim,verbose=True, normalize = False, metricMDS = True, distBeyondNN = False, tconorm = "canonical")
    rs[method] = r

import pickle
with open('./results/results_cifar_no_normalization.pkl', 'wb') as f:
    pickle.dump({'classifier_results': rs}, f)

print("Done!")