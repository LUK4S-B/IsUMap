import os
import sys

# Set the path to the directory containing `isumap.py`
PATH_CURRENT = "../"  # Adjust this path as needed

scriptPath = os.path.abspath(PATH_CURRENT)
sys.path.append(scriptPath)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn import manifold, datasets
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from time import time
from isumap import isumap 
import umap

fig_dir = './../results/Mammoth/'

# Mammoth
mammoth = pd.read_csv('./../../Dataset_files/mammoth.csv')
mammoth_sample = mammoth.sample(20000)
n_neighbors=20


X_train, X_test, y_train, y_test =train_test_split(mammoth_sample, mammoth_sample, 
                     test_size=3500, random_state=42) 


AC = AgglomerativeClustering(n_clusters=11, linkage='ward')
AC.fit(X_train)
labels = AC.labels_
from sklearn.neighbors import KNeighborsClassifier
KN = KNeighborsClassifier(n_neighbors=n_neighbors)
KNN=KN.fit(X_train,labels)
labels2 = KN.predict(mammoth_sample)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(48,35))
ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal')

ax.set_axis_off()
ax.scatter(mammoth_sample['x'], mammoth_sample['y'], mammoth_sample['z'],s=2, c=labels2, cmap=plt.cm.rainbow)
ax.view_init(10, -170)
plt.savefig(fig_dir + f'mammoth.png')
plt.show()

mammoth_array = mammoth_sample.values
mammoth_array.shape

# Parameters for isumap
NN = [30,40]
for n in NN:
    normalize = True
    distBeyondNN = True
    labels = labels2  # Assuming `t` should be `y`
    metricMDS = True

    # Run isumap
    reduced_data_canonical = isumap(mammoth_array, k=n, d=2,
                                    normalize=normalize, distBeyondNN=distBeyondNN, verbose=True,
                                    dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
                                    labels=labels,
                                    initialization="cMDS", metricMDS=metricMDS, sgd_n_epochs=1000, sgd_lr=1e-2,
                                    sgd_batch_size=None,
                                    sgd_max_epochs_no_improvement=100,
                                    sgd_loss='MSE', sgd_saveplots_of_initializations=False, sgd_saveloss=False,
                                    tconorm='canonical')
    reduced_isumap_c = reduced_data_canonical[0]

    # Plot the reduced data
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(reduced_isumap_c[:, 0], reduced_isumap_c[:, 1], s=2, c=labels2, cmap=plt.cm.rainbow)
    plt.savefig(fig_dir + f'reduced_data_isumap_{n}.png')
    plt.close()
    
    
    #Run isomap
    reduced_data_isomap = manifold.Isomap(n_components=2,n_neighbors=n) .fit_transform(mammoth_array)
    
    # Plot the reduced data
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(reduced_data_isomap[:, 0], reduced_data_isomap[:, 1], s=2, c=labels2, cmap=plt.cm.rainbow)
    plt.savefig(fig_dir + f'reduced_data_isomap_{n}.png')
    plt.close()
    
    
    #Run umap
    reduced_umap = umap.UMAP(n_components=2, n_neighbors=n).fit_transform(mammoth_array)
    
    #plot
    fig = plt.figure(figsize=(10,10))
    plt.scatter(reduced_umap[:,0],reduced_umap[:,1], s=2, c=labels2, cmap=plt.cm.rainbow)
    plt.savefig(fig_dir + f'reduced_data_umap_{n}.png')
    plt.close()
    



        
        
	



	
