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
from data_and_plots import plot_data, load_MNIST
from time import time
from isumap import isumap 
import umap


save_dir_isomap = './../MNIST_isomap_csv'
save_dir_umap =  './../MNIST_umap_csv'
save_dir_isumap = './../MNIST_isumap_csv'


# MNIST
train=pd.read_csv('./../../Dataset_files/mnist_train.csv')
test=pd.read_csv('./../../Dataset_files/mnist_test.csv')
y_train = train['label']

X_train = train.drop('label', axis = 1)

y_test = test['label']
X_test = test.drop('label', axis = 1)

X_train['label'] = y_train
X = X_train.sample(n=10000, random_state=42)

y = X['label']
X = X.drop('label', axis = 1)

print(type(X))
print(X.shape)

# Parameters for isumap
D = [400,300,200,150,100,50,30,10,5,2]
for d in D:
    normalize = True
    distBeyondNN = True
    labels = y  
    metricMDS = True

    # Run isumap
    reduced_data_canonical = isumap(X, k=30, d=d,
                                    normalize=normalize, distBeyondNN=distBeyondNN, verbose=True,
                                    dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
                                    labels=labels,
                                    initialization="cMDS", metricMDS=metricMDS, sgd_n_epochs=1000, sgd_lr=1e-2,
                                    sgd_batch_size=None,
                                    sgd_max_epochs_no_improvement=100,
                                    sgd_loss='MSE', sgd_saveplots_of_initializations=False, sgd_saveloss=False,
                                    tconorm='canonical')
    reduced_isumap_c = reduced_data_canonical[0]
    
    # Create DataFrame for reduced data
    reduced_df = pd.DataFrame(reduced_isumap_c, columns=[f'reduced_isumap_d_{i}' for i in range(reduced_isumap_c.shape[1])])
    reduced_df['label'] = y.values

    # save the reduced data
    np.savetxt(os.path.join(save_dir_isumap, f'reduced_isumap_d_{d}.csv'), reduced_isumap_c, delimiter=',')
    reduced_df.to_csv(os.path.join(save_dir_isumap, f'reduced_isumap_labels_d_{d}.csv'), index=False)
    
    #Run umap
    reducer_umap = umap.UMAP(n_neighbors=30, n_components=d, metric='euclidean', random_state=42)
    reduced_data_umap = reducer_umap.fit_transform(X)
    
    
    reduced_df = pd.DataFrame(reduced_data_umap , columns=[f'reduced_umap_d_{i}' for i in range(reduced_data_umap.shape[1])])
    reduced_df['label'] = y.values
    np.savetxt(os.path.join(save_dir_umap, f'reduced_umap_d_{d}.csv'), reduced_data_umap , delimiter=',')
    reduced_df.to_csv(os.path.join(save_dir_umap, f'reduced_umap_labels_d_{d}.csv'), index=False)
     
     #Run Isomap
    reducer_isomap = manifold.Isomap(n_neighbors=30, n_components=d)
    reduced_data_isomap = reducer_isomap.fit_transform(X)
     
    reduced_df = pd.DataFrame(reduced_data_isomap , columns=[f'reduced_isomap_d_{i}' for i in range(reduced_data_isomap.shape[1])])
    reduced_df['label'] = y.values
    np.savetxt(os.path.join(save_dir_isomap, f'reduced_isomap_d_{d}.csv'), reduced_data_isomap, delimiter=',')
    reduced_df.to_csv(os.path.join(save_dir_isomap, f'reduced_isomap_labels_d_{d}.csv'), index=False)
     
     

     
   



        
        
	



	
