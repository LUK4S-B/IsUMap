import pickle
from dimension_reduction_classification import compute_results

import numpy as np
import tensorflow_datasets as tfds
import random
import os

def load_cifar10_data(pickle_file='./../../Dataset_files/cifar10_data.pkl', N=100):
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

ds = [2,3,5,10,20,50,100,200,400]
linearClassifier = True
N = 4000
k = 15
output_dim = 10

methods = ['isumap','umap','isomap']
data, labels = load_cifar10_data(N=2*N)

rs = {}
for method in methods:
    r = compute_results(ds,method,k,linearClassifier,output_dim, data, labels,verbose=True, normalize = False, metricMDS = True, distBeyondNN = False, tconorm = "canonical")
    rs[method] = r

with open('./../results/Classifier_Experiments/results_cifar_no_normalization.pkl', 'wb') as f:
    pickle.dump({'classifier_results': rs}, f)

print("Done!")