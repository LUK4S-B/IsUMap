import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

# import leidenalg as la
# import igraph as ig


# def leiden_clustering(sparse_matrix, weighting_mode = 'undirected'):
#     weighted_graph = ig.Graph.Weighted_Adjacency(sparse_matrix, mode=weighting_mode)
#     partition = la.find_partition(weighted_graph, la.ModularityVertexPartition)
#     # partition = la.find_partition(weighted_graph, la.CPMVertexPartition)
#     return partition

# def leiden_cluster(g, weighting_mode = 'undirected'):
#     leiden_partition = leiden_clustering(g, weighting_mode = weighting_mode)
#     cluster_labels = np.array(leiden_partition.membership)
#     return cluster_labels

def linkage_clustering(dists, method = "average", clusternumber = 10, plot=False):
    # method can be "single", "average", and "complete".
    # Best results were obtained for "average"

    condensed_distances = squareform(dists) 
    # squareform makes distance matrix into a condensed distance matrix, which is required input to linkage clustering algo. A condensed distance matrix is a flat array containing the upper triangular of the distance matrix.
    Z = linkage(condensed_distances, method=method)

    if plot:
        fig = plt.figure(figsize=(25, 20))
        dn = dendrogram(Z)
        plt.show()

    cluster_labels = fcluster(Z, t=clusternumber, criterion='maxclust')

    return Z, cluster_labels

def linkage_cluster(g, clusternumber = 10, plot=False):
    _, cluster_labels = linkage_clustering(g, method = "average", clusternumber = clusternumber, plot=plot)
    return cluster_labels







### --- find optimal clusternumber in linkage clustering --- ###

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import warnings

def find_optimal_clusters_elbow(Z, max_clusters=20):
    """
    Find optimal number of clusters using the elbow method on linkage distances.
    
    The elbow method looks for the "elbow" in the plot of within-cluster distances
    vs number of clusters. The elbow represents the point where adding more clusters
    doesn't significantly improve the clustering quality.
    
    Parameters:
    -----------
    Z : linkage matrix from scipy.cluster.hierarchy.linkage
    max_clusters : int, maximum number of clusters to consider
    
    Returns:
    --------
    optimal_k : int, optimal number of clusters
    distances : list, linkage distances for each merge step
    """
    # Extract linkage distances (heights in dendrogram)
    distances = Z[:, 2]
    
    # Calculate differences between consecutive distances
    # Large jumps indicate natural cluster boundaries
    diffs = np.diff(distances[::-1])  # Reverse to go from many to few clusters
    
    # Find the largest jump (elbow point)
    # Add 2 because we start from the end and need to account for indexing
    optimal_k = np.argmax(diffs) + 2
    
    return min(optimal_k, max_clusters), distances

def find_optimal_clusters_inconsistency(Z, max_clusters=20, depth=2):
    """
    Find optimal clusters using inconsistency coefficient.
    
    The inconsistency coefficient measures how inconsistent a given link is
    compared to the other links at the same level of the hierarchy.
    High inconsistency suggests a natural cluster boundary.
    
    Parameters:
    -----------
    Z : linkage matrix
    max_clusters : int, maximum number of clusters to consider  
    depth : int, depth for inconsistency calculation
    
    Returns:
    --------
    optimal_k : int, optimal number of clusters
    inconsistencies : array, inconsistency coefficients
    """
    # Calculate inconsistency coefficient for each merge
    inconsistencies = inconsistent(Z, d=depth)[:, 3]  # Column 3 is the inconsistency coefficient
    
    # Find merges with high inconsistency (> threshold)
    threshold = np.mean(inconsistencies) + np.std(inconsistencies)
    high_inconsistency_indices = np.where(inconsistencies > threshold)[0]
    
    if len(high_inconsistency_indices) == 0:
        return 2, inconsistencies
    
    # The optimal number of clusters is related to the first high inconsistency
    # Number of items - index of first high inconsistency merge
    n_items = Z.shape[0] + 1
    optimal_k = n_items - high_inconsistency_indices[0]
    
    return min(max(optimal_k, 2), max_clusters), inconsistencies

def find_optimal_clusters_silhouette(dists, Z, max_clusters=20, min_clusters=2):
    """
    Find optimal clusters using silhouette analysis.
    
    Silhouette analysis measures how well each point fits within its assigned cluster
    compared to other clusters. Higher average silhouette score indicates better clustering.
    
    Parameters:
    -----------
    dists : distance matrix (square form)
    Z : linkage matrix
    max_clusters : int, maximum number of clusters to consider
    min_clusters : int, minimum number of clusters to consider
    
    Returns:
    --------
    optimal_k : int, optimal number of clusters
    silhouette_scores : list, silhouette scores for each k
    """
    silhouette_scores = []
    k_range = range(min_clusters, min(max_clusters + 1, dists.shape[0]))
    
    for k in k_range:
        cluster_labels = fcluster(Z, t=k, criterion='maxclust')
        
        # Skip if we have only one cluster or each point is its own cluster
        if len(np.unique(cluster_labels)) < 2 or len(np.unique(cluster_labels)) >= len(cluster_labels):
            silhouette_scores.append(-1)
            continue
            
        # Calculate silhouette score using precomputed distances
        try:
            score = silhouette_score(dists, cluster_labels, metric='precomputed')
            silhouette_scores.append(score)
        except:
            silhouette_scores.append(-1)
    
    if not silhouette_scores or max(silhouette_scores) <= 0:
        return min_clusters, silhouette_scores
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    return optimal_k, silhouette_scores

def find_optimal_clusters_gap_statistic(dists, Z, max_clusters=20, n_refs=10):
    """
    Find optimal clusters using Gap statistic.
    
    The Gap statistic compares the within-cluster dispersion for our data
    with that expected under a null reference distribution (random data).
    The optimal k maximizes the gap between these two dispersions.
    
    Note: This is an approximation since we're working with precomputed distances.
    """
    def calculate_within_cluster_dispersion(distance_matrix, labels):
        """Calculate sum of within-cluster pairwise distances"""
        dispersion = 0
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > 1:
                cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                dispersion += np.sum(cluster_distances) / (2 * len(cluster_indices))
        
        return dispersion
    
    gaps = []
    k_range = range(2, min(max_clusters + 1, dists.shape[0]))
    
    for k in k_range:
        cluster_labels = fcluster(Z, t=k, criterion='maxclust')
        
        # Calculate within-cluster dispersion for actual data
        actual_dispersion = calculate_within_cluster_dispersion(dists, cluster_labels)
        
        # Calculate expected dispersion for random reference data
        ref_dispersions = []
        for _ in range(n_refs):
            # Generate random distance matrix with similar properties
            n = dists.shape[0]
            ref_dists = np.random.uniform(0, np.max(dists), size=(n, n))
            ref_dists = (ref_dists + ref_dists.T) / 2  # Make symmetric
            np.fill_diagonal(ref_dists, 0)  # Zero diagonal
            
            # Cluster the reference data
            ref_condensed = squareform(ref_dists)
            ref_Z = linkage(ref_condensed, method='average')
            ref_labels = fcluster(ref_Z, t=k, criterion='maxclust')
            
            ref_dispersion = calculate_within_cluster_dispersion(ref_dists, ref_labels)
            ref_dispersions.append(np.log(ref_dispersion) if ref_dispersion > 0 else 0)
        
        # Calculate gap
        log_actual = np.log(actual_dispersion) if actual_dispersion > 0 else 0
        expected_log_ref = np.mean(ref_dispersions)
        gap = expected_log_ref - log_actual
        gaps.append(gap)
    
    if not gaps:
        return 2, gaps
    
    # Find the smallest k such that Gap(k) >= Gap(k+1) - std_error(k+1)
    optimal_k = k_range[0]  # Default to minimum
    for i, k in enumerate(k_range[:-1]):
        if i + 1 < len(gaps):
            if gaps[i] >= gaps[i + 1]:
                optimal_k = k
                break
    
    return optimal_k, gaps

def linkage_clustering_auto(dists, method="average", max_clusters=20, 
                           auto_method="silhouette", plot=False, verbose=True):
    """
    Hierarchical clustering with automatic optimal cluster detection.
    
    Parameters:
    -----------
    dists : square distance matrix
    method : str, linkage method ("single", "average", "complete", "ward")
    max_clusters : int, maximum number of clusters to consider
    auto_method : str, method for finding optimal clusters
        - "elbow": Elbow method on linkage distances
        - "inconsistency": Inconsistency coefficient method
        - "silhouette": Silhouette analysis (recommended)
        - "gap": Gap statistic (slower but robust)
        - "all": Try all methods and use majority vote
    plot : bool, whether to plot dendrogram and optimization curves
    verbose : bool, whether to print results
    
    Returns:
    --------
    Z : linkage matrix
    cluster_labels : array, cluster assignments
    optimal_k : int, optimal number of clusters found
    scores : dict, optimization scores for different methods
    """
    
    # Perform hierarchical clustering
    condensed_distances = squareform(dists)
    Z = linkage(condensed_distances, method=method)
    
    scores = {}
    optimal_ks = {}
    
    # Apply selected method(s)
    if auto_method in ["elbow", "all"]:
        k_elbow, distances = find_optimal_clusters_elbow(Z, max_clusters)
        optimal_ks["elbow"] = k_elbow
        scores["elbow_distances"] = distances
        
    if auto_method in ["inconsistency", "all"]:
        k_inconsist, inconsistencies = find_optimal_clusters_inconsistency(Z, max_clusters)
        optimal_ks["inconsistency"] = k_inconsist
        scores["inconsistencies"] = inconsistencies
        
    if auto_method in ["silhouette", "all"]:
        k_silhouette, sil_scores = find_optimal_clusters_silhouette(dists, Z, max_clusters)
        optimal_ks["silhouette"] = k_silhouette
        scores["silhouette_scores"] = sil_scores
        
    if auto_method in ["gap", "all"]:
        k_gap, gap_scores = find_optimal_clusters_gap_statistic(dists, Z, max_clusters)
        optimal_ks["gap"] = k_gap
        scores["gap_scores"] = gap_scores
    
    # Determine final optimal k
    if auto_method == "all":
        # Use majority vote or median
        k_values = list(optimal_ks.values())
        optimal_k = int(np.median(k_values))
        if verbose:
            print("Optimal k by different methods:")
            for method_name, k in optimal_ks.items():
                print(f"  {method_name}: {k}")
            print(f"Final choice (median): {optimal_k}")
    else:
        optimal_k = optimal_ks[auto_method]
        if verbose:
            print(f"Optimal number of clusters ({auto_method} method): {optimal_k}")
    
    # Get final cluster labels
    cluster_labels = fcluster(Z, t=optimal_k, criterion='maxclust')
    
    # Plotting
    if plot:
        if auto_method in ["silhouette", "all"] and "silhouette_scores" in scores:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Dendrogram
            axes[0, 0].set_title("Dendrogram")
            dendrogram(Z, ax=axes[0, 0])
            axes[0, 0].axhline(y=Z[-optimal_k+1, 2], color='r', linestyle='--', 
                              label=f'Cut at {optimal_k} clusters')
            axes[0, 0].legend()
            
            # Silhouette scores
            if "silhouette_scores" in scores:
                k_range = range(2, len(scores["silhouette_scores"]) + 2)
                axes[0, 1].plot(k_range, scores["silhouette_scores"], 'bo-')
                axes[0, 1].axvline(x=optimal_ks.get("silhouette", optimal_k), color='r', linestyle='--')
                axes[0, 1].set_title("Silhouette Analysis")
                axes[0, 1].set_xlabel("Number of Clusters")
                axes[0, 1].set_ylabel("Silhouette Score")
                axes[0, 1].grid(True, alpha=0.3)
            
            # Elbow plot
            if "elbow_distances" in scores:
                axes[1, 0].plot(range(1, len(scores["elbow_distances"]) + 1), 
                               scores["elbow_distances"], 'go-')
                axes[1, 0].set_title("Elbow Method")
                axes[1, 0].set_xlabel("Merge Step")
                axes[1, 0].set_ylabel("Linkage Distance")
                axes[1, 0].grid(True, alpha=0.3)
            
            # Gap statistic
            if "gap_scores" in scores:
                k_range = range(2, len(scores["gap_scores"]) + 2)
                axes[1, 1].plot(k_range, scores["gap_scores"], 'mo-')
                axes[1, 1].axvline(x=optimal_ks.get("gap", optimal_k), color='r', linestyle='--')
                axes[1, 1].set_title("Gap Statistic")
                axes[1, 1].set_xlabel("Number of Clusters")
                axes[1, 1].set_ylabel("Gap")
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        else:
            # Simple dendrogram plot
            plt.figure(figsize=(12, 8))
            dendrogram(Z)
            plt.axhline(y=Z[-optimal_k+1, 2], color='r', linestyle='--', 
                       label=f'Cut at {optimal_k} clusters')
            plt.title("Dendrogram with Optimal Cut")
            plt.legend()
            plt.show()
    
    return Z, cluster_labels, optimal_k, scores

# Convenience function matching your original interface
def linkage_cluster_auto(g, method="average", auto_method="silhouette", 
                        max_clusters=20, plot=False):
    """
    Simplified interface for automatic clustering.
    
    Parameters:
    -----------
    g : square distance matrix
    method : str, linkage method
    auto_method : str, method for finding optimal clusters
    max_clusters : int, maximum clusters to consider
    plot : bool, whether to plot results
    
    Returns:
    --------
    cluster_labels : array, cluster assignments
    """
    _, cluster_labels, optimal_k, _ = linkage_clustering_auto(
        g, method=method, auto_method=auto_method, 
        max_clusters=max_clusters, plot=plot
    )
    return cluster_labels

# Example usage
if __name__ == "__main__":
    # Generate example distance matrix
    np.random.seed(42)
    n_points = 50
    
    # Create some structure: 3 main clusters
    cluster1 = np.random.normal(0, 0.5, (15, 2))
    cluster2 = np.random.normal([3, 0], 0.5, (20, 2))
    cluster3 = np.random.normal([0, 3], 0.5, (15, 2))
    
    points = np.vstack([cluster1, cluster2, cluster3])
    
    # Calculate distance matrix
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(points, metric='euclidean'))
    
    print("Testing automatic cluster detection methods:")
    print("=" * 50)
    
    # Test different methods
    methods = ["silhouette", "elbow", "inconsistency", "all"]
    
    for method in methods:
        print(f"\nTesting {method} method:")
        Z, labels, k, scores = linkage_clustering_auto(
            distances, auto_method=method, plot=False, verbose=True
        )
        print(f"Unique clusters found: {len(np.unique(labels))}")
    
    # Show detailed analysis with plots
    print("\nDetailed analysis with plots:")
    Z, labels, k, scores = linkage_clustering_auto(
        distances, auto_method="all", plot=True, verbose=True
    )