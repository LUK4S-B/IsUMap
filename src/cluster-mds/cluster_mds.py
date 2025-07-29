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

# from cluster_sgd_simplified import optimize_cluster_separation
from clusterSeparationOptimizer import optimize_cluster_separation

def dijkstra_wrapper(graph, i):
    return dijkstra(csgraph=graph, indices=i, directed=False, return_predecessors=False)

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

def cluster_mds(g, cluster_algo, geodesic = True, d = 2, verbose = True, phi_inv = identity, sep_parameter = 1.1, true_labels = None, global_embedding = True):
    N = g.shape[0]
    cluster_labels = cluster_algo(g)
    # clusters = [g[cluster_labels==a][cluster_labels==a] for a in np.unique(cluster_labels)]
    unique_cluster_labels = np.unique(cluster_labels)
    # cluster_number = len(unique_cluster_labels)
    cluster_embeddings = []
    cluster_indices = []
    cluster_medoids = []
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
            partial_func = partial(dijkstra_wrapper, cluster)
            D = []
            if __name__ == 'cluster_mds_sgd_global':
                with Pool() as p:
                    D = p.map(partial_func, range(cluster.shape[0]))
                    p.close()
                    p.join()
            cluster = np.array(D)
            t1 = time()
            if verbose:
                printtime("Dijkstra",t1-t0)
        cluster_medoid = np.argmin(np.sum(cluster, axis=1)) # compute medoid (https://en.wikipedia.org/wiki/Medoid), similar to geometric median (https://en.wikipedia.org/wiki/Geometric_median#Definition) or Fermat point (https://en.wikipedia.org/wiki/Fermat_point)
        cluster_medoids.append(cluster_medoid)

        if not global_embedding:
            cluster_embedding = classical_multidimensional_scaling(cluster, d, verbose)
            cluster_embeddings.append(cluster_embedding)

    if global_embedding:
        if not geodesic:
            g = g.todense()
        emb = classical_multidimensional_scaling(g, d, verbose)
        medoid_list = []
        for cluster_index, cluster_medoid in zip(cluster_indices, cluster_medoids):
            low_dim_cluster = emb[cluster_index]
            cluster_embeddings.append(low_dim_cluster)
            medoid_list.append(low_dim_cluster[cluster_medoid])

    cluster_medoids = np.array(cluster_medoids)

    if geodesic:
        medoid_distances = g[cluster_medoids][:,cluster_medoids]
    else:
        geodesic_medoid_distances = dijkstra(g,
                        directed=False,
                        indices=cluster_medoids,
                        return_predecessors=False)
        # Extract submatrix for select points only
        medoid_distances = geodesic_medoid_distances[:, cluster_medoids]

    medoid_distances = np.array(medoid_distances)
    
    if not global_embedding:
        medoid_embedding = classical_multidimensional_scaling(medoid_distances, d, verbose)
        medoid_list = [medoid for medoid in medoid_embedding]

    for medoid,cemb in zip(medoid_list,cluster_embeddings):
        matches = np.all(cemb == medoid, axis=1)
        yes = np.any(matches)
        print(yes)

    # plot_data(medoid_embedding, unique_cluster_labels, title="medoids", display=True, save=True, size=10)

    # complete_embedding = np.zeros([N, d])
    complete_embeddings = []
    c = 0
    if global_embedding:
        for cluster_embedding in cluster_embeddings:
            complete_embeddings.append(cluster_embedding)
    else:
        for cluster_embedding, cluster_index, medoid in zip(cluster_embeddings, cluster_indices, medoid_embedding):
            # complete_embedding[cluster_index] = cluster_embedding + medoid
            complete_embeddings.append(cluster_embedding + medoid)


    if true_labels is not None:
        true_labels_reordered_array = np.concatenate(true_labels_reordered)
    else:
        true_labels_reordered_array = None
        # c+=1
        # plot_data(complete_embedding, cluster_labels, title="complete_"+str(c), display=True, save=False)
    
    # plot_data(complete_embedding, cluster_labels, title="complete_" + str(sep_parameter), display=True, save=True)
    # if true_labels is not None:
    #     plot_data(complete_embedding, true_labels, title="complete true_" + str(sep_parameter), display=True, save=True)
    # special_plot(emb,medoid_list,cluster_labels,unique_cluster_labels)

    print("Starting optimization...")

    results = optimize_cluster_separation(complete_embeddings, medoid_list, medoid_distances, labels=true_labels_reordered)
    # results = optimize_cluster_separation(complete_embeddings, medoid_list, medoid_distances)

    # Extract the optimizer model from results
    sgd_optimized_model = results['optimizer_model']
    medoid_paths = results['medoid_paths']
    labels = results['labels']
    losses = results['losses']

    transformed_clusters = sgd_optimized_model.apply_transformations()
    current_medoids = sgd_optimized_model.get_current_medoids()

    transformed_clusters = np.vstack([np.array(cluster.detach().numpy()) for cluster in transformed_clusters])
    
    # Store results to file
    import pickle
    import os
    
    # Create results directory if it doesn't exist
    results_dir = "optimization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results_dir}/cluster_mds_results_{timestamp}.pkl"
    
    # Prepare data for saving (convert tensors to numpy arrays)
    save_data = {
        'transformed_clusters': transformed_clusters,
        'medoid_paths': medoid_paths,
        'labels': labels,
        'losses': losses,
        'cluster_labels': cluster_labels,
        'true_labels_reordered_array': true_labels_reordered_array,
        'sep_parameter': sep_parameter,
        'optimizer_model_state': sgd_optimized_model.state_dict()
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Results saved to: {filename}")
    
    plot_data(transformed_clusters, np.sort(cluster_labels), title="complete_" + str(sep_parameter), display=True, save=True)
    if true_labels is not None:
        plot_data(transformed_clusters, true_labels_reordered_array, title="complete true_" + str(sep_parameter), display=True, save=True)




