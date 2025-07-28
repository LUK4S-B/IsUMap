import os
import sys

# Set the path to the directory containing `isumap.py`
PATH_CURRENT =  "../../src/"  # Adjust this path as needed
fig_dir ="./../results/BreastCancer/"

scriptPath = os.path.abspath(PATH_CURRENT)
sys.path.append(scriptPath)

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn import manifold, datasets
import seaborn as sns
from isumap import isumap
import umap

k=30
data=pd.read_csv('./../../Dataset_files/BreastCancerDataset.csv')
data = data.drop('id',axis=1)
data=data.drop('Unnamed: 32',axis=1)
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
X = data.values
colors=data['diagnosis']

# IsUMAP function
def Isu(data, labels, k, dim, normalize=True, 
        metricMDS=True, distBeyondNN=True, tconorm="canonical"):
    print('IsUMAP is Running...')
    reduced_data_canonical = isumap(data, k=k, d=dim,
                                    normalize=normalize, distBeyondNN=distBeyondNN, verbose=True,
                                    dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
                                    labels=labels,
                                    initialization="cMDS", metricMDS=metricMDS, sgd_n_epochs=1000, sgd_lr=1e-2,
                                    sgd_batch_size=None,
                                    sgd_max_epochs_no_improvement=100,
                                    sgd_loss='MSE', sgd_saveplots_of_initializations=False, sgd_saveloss=False,
                                    tconorm=tconorm)
    return reduced_data_canonical[0]

# Isomap function
def Iso(data, k, dim):
    print('Isomap is running...')
    reduced_isomap = manifold.Isomap(n_components=dim, n_neighbors=k).fit_transform(data)
    return reduced_isomap

# UMAP function
def UM(data, k, dim):
    print('UMAP is running...')
    reduced_UMAP = umap.UMAP(n_components=dim, n_neighbors=k).fit_transform(data)
    return reduced_UMAP

#Scatter plot function
def S_plot(data, labels, title, fig_dir, file_name):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=colors, cmap=plt.cm.rainbow, s=10, alpha=0.7, edgecolors='k')
    plt.title(title)
    plt.savefig(os.path.join(fig_dir, file_name))
    plt.show()


# Run IsUMAP
isumap_result = Isu(X, colors, k=k, dim=2)

# Run Isomap
isomap_result = Iso(X, k=k, dim=2)

# Run UMAP
umap_result = UM(X, k=k, dim=2)

# Plot the results using the scatter plot function
S_plot(isumap_result, colors, "IsUMAP", fig_dir, "isumap_result.png")
S_plot(isomap_result, colors, "Isomap", fig_dir, "isomap_result.png")
S_plot(umap_result, colors, "UMAP", fig_dir, "umap_result.png")

