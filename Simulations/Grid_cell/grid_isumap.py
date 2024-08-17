from utils import *
import matplotlib.pyplot as plt
import h5py
import numpy as np
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from datetime import datetime
import time
import os
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from persim import plot_diagrams
from isumap import isumap

# Custom function to plot barcode
def custom_plot_barcode(diagrams, title="Barcode"):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Homology group")

    # Plot barcodes for each dimension
    for dim, diagram in enumerate(diagrams):
        for bar in diagram:
            ax.plot([bar[0], bar[1]], [dim, dim], lw=2)

    return fig, ax

fig_dir = ":/../results/Grid_cell"
rat_name = 'R'
mod_name = '1'
sess_name = 'OF'
day_name = 'day1'

sspikes = get_spikes(rat_name, mod_name, day_name, sess_name, bType='pure',
                     bSmooth=True, bSpeed=True)[0]

bRoll = False
dim = 6
ph_classes = [0, 1]  # Decode the ith most persistent cohomology class
num_circ = len(ph_classes)
dec_tresh = 0.99
metric = 'cosine'
maxdim = 1
coeff = 47
active_times = 15000
k = 1000
num_times = 5
n_points = 1200
nbs = 800
sigma = 1500
folder = '/net/st1/export/clusterhome/fahimi/Documents/Grid_cell/'

# Iterates over the different datasets
for rat_name, mod_name, sess_name, day_name in [('R', '1', 'OF', 'day1')]:
    if sess_name in ('OF', 'WW'):
        sspikes, __, __, __, __ = get_spikes(rat_name, mod_name, day_name, sess_name, bType='pure',
                                             bSmooth=True, bSpeed=True, folder=folder)
    else:
        sspikes = get_spikes(rat_name, mod_name, day_name, sess_name, bType='pure', bSmooth=True, bSpeed=False, folder=folder)
        
    num_neurons = len(sspikes[0, :])

    if bRoll:
        np.random.seed(s)
        shift = np.zeros(num_neurons, dtype=int)
        for n in range(num_neurons):
            shifti = int(np.random.rand() * len(sspikes[:, 0]))
            sspikes[:, n] = np.roll(sspikes[:, n].copy(), shifti)
            shift[n] = shifti
            
    times_cube = np.arange(0, len(sspikes[:, 0]), num_times)
    movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube, :], 1))[-active_times:])
    movetimes = times_cube[movetimes]
    
    
    dim_red_spikes_move_scaled,__,__ = pca(preprocessing.scale(sspikes[movetimes,:]), dim = dim)
    indstemp,dd,fs  = sample_denoising(dim_red_spikes_move_scaled,  k, 
                                        n_points, 1, metric)
    dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]
    #X = squareform(pdist(dim_red_spikes_move_scaled, metric))
    
    #pre_red_spikes_move = preprocessing.scale(sspikes[movetimes,:])
    #print('pre_red_spikes_move', pre_red_spikes_move.shape)
    #indstemp, dd, fs = sample_denoising(pre_red_spikes_move, k, n_points, 1, metric)
    #pre_red_spikes_move = pre_red_spikes_move[indstemp, :]

    normalize = True
    distBeyondNN = True
    labels = None
    metricMDS = True

    # Run isumap
    reduced_data_canonical = isumap(dim_red_spikes_move_scaled, k=15, d=6,
                                    normalize=normalize, distBeyondNN=distBeyondNN, verbose=True,
                                    dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
                                    labels=labels,
                                    initialization="cMDS", metricMDS=metricMDS, sgd_n_epochs=1000, sgd_lr=1e-2,
                                    sgd_batch_size=None,
                                    sgd_max_epochs_no_improvement=100,
                                    sgd_loss='MSE', sgd_saveplots_of_initializations=False, sgd_saveloss=False,
                                    tconorm='canonical')
    reduced_isumap_3 = reduced_data_canonical[1]
    #D_3 = reduced_data_canonical[0]
   

    # Plot 3D embedding
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(reduced_isumap_3[:, 0], reduced_isumap_3[:, 1], reduced_isumap_3[:, 2], s=5)
    ax.view_init(azim=100, elev=320)

    # Save the 3D embedding plot
    plot_path = os.path.join(fig_dir, f"{rat_name}_{mod_name}_{sess_name}_{day_name}_3D_embedding.png")
    plt.savefig(plot_path)
    plt.close()

    # Calculate persistence
    persistence = ripser(reduced_isumap_3 , maxdim=maxdim, coeff=coeff, do_cocycles=True, distance_matrix=False, metric=metric)

    # Plot and save persistence diagrams
    persistence_path = os.path.join(fig_dir, f"{rat_name}_{mod_name}_{sess_name}_{day_name}_persistence_diagrams.png")
    plot_diagrams(persistence['dgms'])
    plt.savefig(persistence_path)
    plt.close()

    # Plot and save barcode using custom function
    barcode_path = os.path.join(fig_dir, f"{rat_name}_{mod_name}_{sess_name}_{day_name}_barcode.png")
    fig, ax = custom_plot_barcode(persistence['dgms'])
    fig.savefig(barcode_path)
    plt.close(fig)

    if len(day_name) > 0:
        day_name = '_' + day_name
    print(rat_name + '_' + mod_name + '_' + sess_name + day_name)

    plt.show()  # This will display all the figures if running interactively
