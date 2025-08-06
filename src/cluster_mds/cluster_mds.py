import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_CURRENT = os.path.join(SCRIPT_DIR, "../")
scriptPath = os.path.abspath(PATH_CURRENT)
sys.path.append(scriptPath)

import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix, lil_matrix, save_npz
from time import time

from dimension_reduction_schemes import classical_multidimensional_scaling
from data_and_plots import plot_data, printtime

from clusterSeparationOptimizer import optimize_cluster_separation

def dijkstra_wrapper(graph, directedDistances, i):
    return dijkstra(csgraph=graph, indices=i, directed=directedDistances, return_predecessors=False)

def identity(x):
    return x

def convert_graph_to_dist_matrix(g, phi_inv):
    if not isinstance(g, lil_matrix):
        g = g.tolil()
    if phi_inv != identity:
        for i, (row_indices, row_data) in enumerate(zip(g.rows, g.data)):
            for j, value in zip(row_indices, row_data):
                g[i, j] = phi_inv(value) + np.nextafter(0., np.float32(1.))
    return g.tocsr()

import matplotlib.pyplot as plt

def special_plot(data_array,special_points,colors,special_point_colors):
    plt.figure(figsize=(10, 8))

    # Plot the main array data as scatter points
    plt.scatter(data_array[:, 0], data_array[:, 1], 
            c=colors, cmap="tab10", alpha=0.6, s=50, label='Main data')

    special_points_array = np.array(special_points)
    special_point_colors = np.array(special_point_colors)

    # Plot the special points as stars with different colors
    scatter = plt.scatter(special_points_array[:, 0], special_points_array[:, 1], c=special_point_colors, marker='*', s=200, cmap='tab10', edgecolors='black', linewidth=1, label='Special points')

    # Customize the plot
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Array Data with Special Star Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # Equal aspect ratio

    # Show the plot
    plt.tight_layout()
    plt.show()

def cluster_mds(g, cluster_algo, geodesic = True, d = 2, verbose = True, phi_inv = identity, true_labels = None, global_embedding = True, directedDistances = False, store_results = False, display_results = False, save_display_results = True, plot_title = "title", also_return_optimizer_model = False, also_return_medoid_paths = False, orig_data = None, plot_original_data = False, visualize_results = False, custom_color_map='jet', **cluster_algo_kwargs):

    cluster_labels = cluster_algo(g, **cluster_algo_kwargs)

    if orig_data is not None and plot_original_data:
        if true_labels is not None:
            plot_data(orig_data, true_labels, title="Initial dataset with true labels",  display=True, save=False)
        if cluster_labels is not None:
            plot_data(orig_data, cluster_labels, title="Initial dataset with cluster labels",  display=True, save=False)

    unique_cluster_labels = np.unique(cluster_labels)
    if verbose:
        print("Number of clusters: ", str(len(unique_cluster_labels)))
    cluster_embeddings = []
    cluster_indices = []
    cluster_medoid_indices = []
    true_labels_reordered = []
    if not geodesic:
        g = convert_graph_to_dist_matrix(g, phi_inv)
        
    for label in unique_cluster_labels:
        cluster_index = cluster_labels==label
        cluster_indices.append(cluster_index)
        cluster = g[cluster_index][:,cluster_index]
        if true_labels is not None:
            true_labels_reordered.append(true_labels[cluster_index])
        if not geodesic:
            # apply dijkstra to cluster
            if verbose:
                print("\nRunning Dijkstra...")
            t0 = time()
            partial_func = partial(dijkstra_wrapper, cluster, directedDistances)
            D = []
            if __name__ == 'cluster_mds':
                with Pool() as p:
                    D = p.map(partial_func, range(cluster.shape[0]))
                    p.close()
                    p.join()
            cluster = np.array(D)
            t1 = time()
            if verbose:
                printtime("Dijkstra",t1-t0)
        cluster_medoid_index = np.argmin(np.sum(cluster, axis=1)) # compute medoid (https://en.wikipedia.org/wiki/Medoid), similar to geometric median (https://en.wikipedia.org/wiki/Geometric_median#Definition) or Fermat point (https://en.wikipedia.org/wiki/Fermat_point)
        cluster_medoid_indices.append(cluster_medoid_index)

        if not global_embedding:
            cluster_embedding = classical_multidimensional_scaling(cluster, d, verbose)
            cluster_embeddings.append(cluster_embedding)

    # for i in range(len(cluster_indices)):
    #     print(cluster_indices[i][cluster_indices[i]==True].shape)

    cluster_medoid_indices = np.array(cluster_medoid_indices)
    # Convert cluster medoid indices to global indices
    global_medoid_indices = []
    current_index = 0
    for i, cluster_index in enumerate(cluster_indices):
        cluster_size = np.sum(cluster_index)
        global_medoid_index = current_index + cluster_medoid_indices[i]
        global_medoid_indices.append(global_medoid_index)
        current_index += cluster_size
    global_medoid_indices = np.array(global_medoid_indices)

    if geodesic:
        medoid_distances = g[global_medoid_indices][:, global_medoid_indices]
    else:
        geodesic_medoid_distances = dijkstra(g,
                        directed=directedDistances,
                        indices=global_medoid_indices,
                        return_predecessors=False)
        # Extract submatrix for select points only
        medoid_distances = geodesic_medoid_distances[:, global_medoid_indices]

    medoid_distances = np.array(medoid_distances)

    complete_embeddings = []
    if global_embedding:
        if not isinstance(g, np.ndarray):
            g = g.todense()
        emb = classical_multidimensional_scaling(g, d, verbose)
        medoid_list = []
        for cluster_index, cluster_medoid in zip(cluster_indices, cluster_medoid_indices):
            low_dim_cluster = emb[cluster_index]
            cluster_embeddings.append(low_dim_cluster)
            medoid_list.append(low_dim_cluster[cluster_medoid])
        cluster_medoid_indices = np.array(cluster_medoid_indices)

        for cluster_embedding in cluster_embeddings:
            complete_embeddings.append(cluster_embedding)
    else:
        medoid_embedding = classical_multidimensional_scaling(medoid_distances, d, verbose)
        medoid_list = [medoid for medoid in medoid_embedding]
        for cluster_embedding, cluster_index, medoid in zip(cluster_embeddings, cluster_indices, medoid_embedding):
            # complete_embedding[cluster_index] = cluster_embedding + medoid
            complete_embeddings.append(cluster_embedding + medoid)

    if true_labels is not None:
        true_labels_reordered_array = np.concatenate(true_labels_reordered)
    else:
        true_labels_reordered = None
        true_labels_reordered_array = None


    # --------------- CLUSTER SEPARATION OPTIMIZATION ---------------
    print("\nStarting cluster separation optimization...")
    results = optimize_cluster_separation(complete_embeddings, medoid_list, medoid_distances, labels=true_labels_reordered, visualize=visualize_results, custom_color_map=custom_color_map)

    # Organize data
    sgd_optimized_model = results['optimizer_model']
    medoid_paths = results['medoid_paths']
    losses = results['losses']

    transformed_clusters = sgd_optimized_model.apply_transformations()
    current_medoids = sgd_optimized_model.get_current_medoids()

    transformed_clusters = np.vstack([np.array(cluster.detach().numpy()) for cluster in transformed_clusters])

    return_data = {
        'transformed_clusters': transformed_clusters,
        'cluster_indices': cluster_indices,
        'current_medoids': current_medoids,
        'cluster_labels': np.sort(cluster_labels),
        'true_labels': true_labels_reordered_array,
        'true_labels_split_into_clusters': true_labels_reordered,
        'losses': losses,
        }
    if also_return_optimizer_model:
        return_data['optimizer_model'] = sgd_optimized_model
    if also_return_medoid_paths:
        return_data['medoid_paths'] = medoid_paths

    if store_results:
        import pickle
        import os

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/cluster_mds_results_{timestamp}.pkl"
        
        results_dir = "optimization_results"
        os.makedirs(results_dir, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(return_data, f)
        
        print(f"Results saved to: {filename}")
    
    if display_results:
        plot_data(transformed_clusters, np.sort(cluster_labels), title=plot_title, display=True, save=save_display_results)
        if true_labels is not None:
            plot_data(transformed_clusters, true_labels_reordered_array, title=plot_title + " true_labels ", display=True, save=save_display_results)

    return return_data


