import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')
import copy
import os
from PIL import Image
import io

class ClusterSeparationOptimizer(nn.Module):
    """
    PyTorch module for separating clusters using rotations and translations
    such that no cluster points lie within other clusters' convex hulls.
    Optimized vectorized version with improved point-in-hull checking.
    """
    
    def __init__(self, clusters: List[np.ndarray], medoids: List[np.ndarray], medoid_distances: np.ndarray, intercluster_distance_factor = 0.3, rotation_flag: bool = False, separation_weight: float = 1.0, drift_weight: float = 1.0, translation_penalty_factor: float = 0.0, rotation_penalty_factor: float = 1.0):
        """
        Initialize the optimizer with clusters and their medoids.
        
        Args:
            clusters: List of numpy arrays, each of shape (n_i, 2)
            medoids: List of numpy arrays, each of shape (2,) - cluster centers
        """
        super().__init__()

        self.separation_weight = separation_weight
        self.drift_weight = drift_weight
        self.translation_penalty_factor = translation_penalty_factor
        self.rotation_penalty_factor = rotation_penalty_factor
        self.rotation_flag = rotation_flag
        self.n_clusters = len(clusters)

        if len(medoids) != self.n_clusters:
            raise ValueError("Number of medoids must match number of clusters")
        
        # Precompute hull vertices for each cluster
        self.hull_vertices_list = []
        self.max_hull_vertices = 0
        self.cluster_diameters = torch.zeros(self.n_clusters)
        for i, cluster in enumerate(clusters):
            hull = ConvexHull(cluster)
            hull_vertices = torch.tensor(hull.vertices, dtype=torch.long)
            self.hull_vertices_list.append(hull_vertices)
            self.max_hull_vertices = max(self.max_hull_vertices, len(hull_vertices))

            self.cluster_diameters[i] = self.compute_diameter(torch.tensor(cluster[hull_vertices], dtype=torch.float32))

        # Convert to tensors and store original data
        self.original_clusters = [torch.tensor(c, dtype=torch.float32) for c in clusters]
        self.original_medoids = [torch.tensor(m, dtype=torch.float32) for m in medoids]
        
        # Precompute cluster sizes and create padded tensor for efficient operations
        self.cluster_sizes = [len(cluster) for cluster in clusters]
        self.max_cluster_size = max(self.cluster_sizes)
        
        # Create padded cluster tensor for vectorized operations
        self.padded_clusters = torch.zeros(self.n_clusters, self.max_cluster_size, 2)
        self.cluster_masks = torch.zeros(self.n_clusters, self.max_cluster_size, dtype=torch.bool)
        
        for i, cluster in enumerate(self.original_clusters):
            size = self.cluster_sizes[i]
            self.padded_clusters[i, :size] = cluster
            self.cluster_masks[i, :size] = True
        
        # Precompute hull vertices in padded format
        self.padded_hull_vertices = torch.zeros(self.n_clusters, self.max_hull_vertices, 2)
        self.hull_masks = torch.zeros(self.n_clusters, self.max_hull_vertices, dtype=torch.bool)
        
        for i, hull_vertices in enumerate(self.hull_vertices_list):
            hull_size = len(hull_vertices)
            cluster_hull_points = self.original_clusters[i][hull_vertices]
            self.padded_hull_vertices[i, :hull_size] = cluster_hull_points
            self.hull_masks[i, :hull_size] = True
        
        self.stacked_medoids = torch.stack(self.original_medoids)

        # Learnable parameters
        self.rotation_angles = nn.Parameter(torch.zeros(self.n_clusters))
        self.translations = nn.Parameter(torch.zeros(self.n_clusters, 2))
        

        self.medoid_distances = torch.tensor(medoid_distances, dtype=torch.float32)
        # self.medoid_distances = torch.log2(1 + self.medoid_distances)
        self.medoid_distances = self.medoid_distances / torch.max(self.medoid_distances)
        self.medoid_distances = self.medoid_distances * intercluster_distance_factor * torch.max(self.cluster_diameters)

    def compute_diameter(self, points):
        # points: N×d tensor (N points in d dimensions)
        # N, d = points.shape
        
        # Expand dimensions for broadcasting: N×1×d and 1×N×d
        p1 = points.unsqueeze(1)  # N×1×d
        p2 = points.unsqueeze(0)  # 1×N×d
        
        # Compute all pairwise differences: N×N×d
        diffs = p1 - p2
        
        # Compute squared distances: N×N
        squared_distances = torch.sum(diffs**2, dim=2)
        
        # Get maximum distance (diameter)
        diameter = torch.sqrt(torch.max(squared_distances))
        
        return diameter

    def apply_transformations_vectorized(self) -> torch.Tensor:
        """
        Apply rotations and translations vectorized across all clusters.
        
        Returns:
            Padded tensor of shape (n_clusters, max_cluster_size, 2)
        """


        if self.rotation_flag:
            # Center all clusters around their medoids
            centered_clusters = self.padded_clusters - self.stacked_medoids.unsqueeze(1)
            
            # Create rotation matrices for all clusters at once
            cos_angles = torch.cos(self.rotation_angles)  # (n_clusters,)
            sin_angles = torch.sin(self.rotation_angles)  # (n_clusters,)
            
            # Build rotation matrices: (n_clusters, 2, 2)
            rotation_matrices = torch.stack([
                torch.stack([cos_angles, -sin_angles], dim=1),
                torch.stack([sin_angles, cos_angles], dim=1)
            ], dim=2)
            
            # Apply rotations: (n_clusters, max_cluster_size, 2) @ (n_clusters, 2, 2) -> (n_clusters, max_cluster_size, 2)
            rotated_clusters = torch.bmm(centered_clusters, rotation_matrices)
            
            # Apply translations
            new_medoids = self.stacked_medoids + self.translations
            transformed_clusters = rotated_clusters + new_medoids.unsqueeze(1)
        else:
            transformed_clusters = self.padded_clusters + self.translations.unsqueeze(1)
        
        return transformed_clusters
    
    def apply_transformations_vectorized_hulls(self) -> torch.Tensor:
        """
        Apply transformations to hull vertices specifically.
        
        Returns:
            Transformed hull vertices: (n_clusters, max_hull_vertices, 2)
        """

        if self.rotation_flag:
            # Center hull vertices around their medoids
            centered_hulls = self.padded_hull_vertices - self.stacked_medoids.unsqueeze(1)
            
            # Create rotation matrices
            cos_angles = torch.cos(self.rotation_angles)
            sin_angles = torch.sin(self.rotation_angles)
            
            rotation_matrices = torch.stack([
                torch.stack([cos_angles, -sin_angles], dim=1),
                torch.stack([sin_angles, cos_angles], dim=1)
            ], dim=2)
            
            # Apply rotations and translations
            rotated_hulls = torch.bmm(centered_hulls, rotation_matrices)
            
            new_medoids = self.stacked_medoids + self.translations

            transformed_hulls = rotated_hulls + new_medoids.unsqueeze(1)
        else:
            transformed_hulls = self.padded_hull_vertices  + self.translations.unsqueeze(1)
        
        return transformed_hulls
    
    def find_k_closest_points_vectorized(self, points: torch.Tensor, target_points: torch.Tensor, k: int = 2000):
        """
        Find k closest points for each cluster to each target point vectorized.
        
        Args:
            points: (n_clusters, max_cluster_size, 2) - all cluster points
            target_points: (n_clusters, 2) - target points (medoids)
            k: number of closest points to find
            
        Returns:
            indices: (n_clusters, n_clusters, k) - indices of closest points
        """
        # Compute pairwise distances: cluster i points to cluster j medoid
        # points: (n_clusters, max_cluster_size, 2)
        # target_points: (n_clusters, 2)
        
        # Expand dimensions for broadcasting
        points_expanded = points.unsqueeze(1)  # (n_clusters, 1, max_cluster_size, 2)
        targets_expanded = target_points.unsqueeze(0).unsqueeze(2)  # (1, n_clusters, 1, 2)
        
        # Compute squared distances: (n_clusters, n_clusters, max_cluster_size)
        distances_sq = torch.sum((points_expanded - targets_expanded) ** 2, dim=3)
        
        # Set distances for padded points to infinity
        mask_expanded = self.cluster_masks.unsqueeze(1)  # (n_clusters, 1, max_cluster_size)
        distances_sq = distances_sq.masked_fill(~mask_expanded, float('inf'))
        
        # Find k smallest distances for each (source_cluster, target_cluster) pair
        k_clamped = min(k, self.max_cluster_size)
        _, indices = torch.topk(distances_sq, k_clamped, dim=2, largest=False)
        
        return indices
    
    def point_in_convex_hull_vectorized_fast(self, points: torch.Tensor, hull_vertices: torch.Tensor, hull_mask: torch.Tensor, i, j, eps: float = 1e-8, visualize_distances: bool = False, ax=None) -> torch.Tensor:
        """
        Fast vectorized version of point-in-convex-hull check with differentiable violation scoring.
        Now correctly computes distances to finite line segments for points outside the hull.
        
        Args:
            points: (batch_size, 2) - points to check
            hull_vertices: (max_hull_vertices, 2) - hull vertices (padded)
            hull_mask: (max_hull_vertices,) - mask for valid hull vertices
            eps: small epsilon for numerical stability
            visualize_distances: Whether to visualize the minimum distance lines
            ax: Matplotlib axis for visualization (if visualize_distances is True)
            
        Returns:
            violations: (batch_size,) - violation scores (0 if outside, sigmoid(min_distance) if inside)
        """
        if hull_mask.sum() < 3:
            return torch.zeros(points.shape[0])
        
        # Get valid hull vertices
        valid_hull = hull_vertices[hull_mask]  # (n_valid_vertices, 2)
        n_vertices = valid_hull.shape[0]
        
        # Create edge vectors: (n_vertices, 2)
        p1 = valid_hull  # (n_vertices, 2)
        p2 = torch.roll(valid_hull, -1, dims=0)  # (n_vertices, 2) - next vertex
        edge_vecs = p2 - p1  # (n_vertices, 2)
        
        # Expand dimensions for broadcasting
        points_exp = points.unsqueeze(1)  # (batch_size, 1, 2)
        p1_exp = p1.unsqueeze(0)  # (1, n_vertices, 2)
        p2_exp = p2.unsqueeze(0)  # (1, n_vertices, 2)
        edge_vecs_exp = edge_vecs.unsqueeze(0)  # (1, n_vertices, 2)
        
        # Compute point vectors from each edge start
        point_vecs = points_exp - p1_exp  # (batch_size, n_vertices, 2)
        
        # Compute cross products (signed distances to infinite lines)
        cross_products = (edge_vecs_exp[:, :, 0] * point_vecs[:, :, 1] - 
                         edge_vecs_exp[:, :, 1] * point_vecs[:, :, 0])  # (batch_size, n_vertices)
        
        # Compute edge lengths for normalization
        edge_lengths = torch.norm(edge_vecs, dim=1) + eps  # (n_vertices,)
        
        # Normalize cross products to get actual signed distances to infinite lines
        signed_distances_to_lines = cross_products / edge_lengths.unsqueeze(0)  # (batch_size, n_vertices)
        
        # Vectorized sign checking: point is inside if all signed distances have same sign
        all_positive = torch.all(signed_distances_to_lines >= -eps, dim=1)  # (batch_size,)
        all_negative = torch.all(signed_distances_to_lines <= eps, dim=1)   # (batch_size,)
        
        # Point is inside if all signs are consistent (all positive OR all negative)
        is_inside = all_positive | all_negative  # (batch_size,)
        
        # For points inside, the minimum distance to infinite lines is correct
        # For points outside, we need to compute distances to finite line segments
        min_distances = torch.zeros(points.shape[0], device=points.device)
        
        # For inside points, use the minimum distance to infinite lines
        inside_mask = is_inside
        if inside_mask.any():
            min_distances[inside_mask] = torch.min(torch.abs(signed_distances_to_lines[inside_mask]), dim=1)[0]
        
        # For outside points, compute distances to finite line segments
        outside_mask = ~is_inside
        if outside_mask.any():
            outside_points = points[outside_mask]  # (n_outside, 2)
            
            # Compute distances to all vertices first
            # (n_outside, 1, 2) - (1, n_vertices, 2) -> (n_outside, n_vertices, 2)
            distances_to_vertices = torch.norm(
                outside_points.unsqueeze(1) - valid_hull.unsqueeze(0), dim=2
            )  # (n_outside, n_vertices)
            
            # Compute distances to finite line segments
            # Project point onto each edge line
            # t = dot(point_vec, edge_vec) / dot(edge_vec, edge_vec)
            edge_lengths_sq = edge_lengths ** 2  # (n_vertices,)
            
            # (n_outside, n_vertices, 2) @ (n_vertices, 2) -> (n_outside, n_vertices)
            dot_products = torch.sum(
                point_vecs[outside_mask] * edge_vecs_exp[0], dim=2
            )  # (n_outside, n_vertices)
            
            # Projection parameter t
            t = dot_products / edge_lengths_sq.unsqueeze(0)  # (n_outside, n_vertices)
            
            # Clamp t to [0, 1] to get projection onto finite line segment
            t_clamped = torch.clamp(t, 0, 1)  # (n_outside, n_vertices)
            
            # Compute projected points
            # p1 + t * edge_vec
            projected_points = p1_exp[0] + t_clamped.unsqueeze(2) * edge_vecs_exp[0]  # (n_outside, n_vertices, 2)
            
            # Compute distances to projected points
            distances_to_edges = torch.norm(
                outside_points.unsqueeze(1) - projected_points, dim=2
            )  # (n_outside, n_vertices)
            
            # For each outside point, take minimum of vertex distances and edge distances
            min_vertex_distances = torch.min(distances_to_vertices, dim=1)[0]  # (n_outside,)
            min_edge_distances = torch.min(distances_to_edges, dim=1)[0]  # (n_outside,)
            
            # Take the minimum of vertex and edge distances
            outside_min_distances = torch.minimum(min_vertex_distances, min_edge_distances)
            
            # Assign to the result tensor
            min_distances[outside_mask] = outside_min_distances
        
        # Apply violation scoring
        violations = torch.where(
            is_inside,
            torch.tanh((self.medoid_distances[i,j] + min_distances)),  # Violation score for inside points
            torch.tanh(torch.max(torch.tensor(0), self.medoid_distances[i,j] - min_distances)) # violation proportional to self.medoid_distances[i,j] - min_distances up to self.medoid_distances[i,j]
        )
        
        # Optional visualization of minimum distance lines
        if visualize_distances and ax is not None:
            self._visualize_distance_lines(points, valid_hull, min_distances, is_inside, ax, projected_points)
        
        return violations
    
    def _visualize_distance_lines(self, points: torch.Tensor, valid_hull: torch.Tensor, min_distances: torch.Tensor, is_inside: torch.Tensor, ax, projected_points: torch.Tensor | None = None) -> None:
        """
        Visualize the minimum distance lines from points to the convex hull.
        
        Args:
            points: (batch_size, 2) - points to check
            valid_hull: (n_vertices, 2) - valid hull vertices
            min_distances: (batch_size,) - minimum distances to hull
            is_inside: (batch_size,) - boolean mask for inside points
            ax: Matplotlib axis for plotting
        """
        import matplotlib.pyplot as plt
        
        # Convert to numpy for plotting
        points_np = points.detach().cpu().numpy()
        hull_np = valid_hull.detach().cpu().numpy()
        min_distances_np = min_distances.detach().cpu().numpy()
        is_inside_np = is_inside.detach().cpu().numpy()
        
        # Plot hull
        hull_closed = np.vstack([hull_np, hull_np[0]])  # Close the hull
        ax.plot(hull_closed[:, 0], hull_closed[:, 1], 'b-', linewidth=2, label='Convex Hull')
        ax.fill(hull_np[:, 0], hull_np[:, 1], alpha=0.1, color='blue')
        
        # Plot points with different colors for inside/outside
        inside_points = points_np[is_inside_np]
        outside_points = points_np[~is_inside_np]
        
        if len(inside_points) > 0:
            ax.scatter(inside_points[:, 0], inside_points[:, 1], c='green', s=50, alpha=0.7, label='Inside Points')
        if len(outside_points) > 0:
            ax.scatter(outside_points[:, 0], outside_points[:, 1], c='red', s=50, alpha=0.7, label='Outside Points')
        
        # For outside points, find and plot the closest points on the hull
        if len(outside_points) > 0 and projected_points is not None:
            # Convert projected_points to numpy
            projected_points_np = projected_points.detach().cpu().numpy()
            
            # For each outside point, find which edge gives the minimum distance
            for i, point in enumerate(outside_points):
                # Find the closest edge for this point
                # We need to compute distances to all projected points for this outside point
                point_idx = np.where(~is_inside_np)[0][i]  # Get the original index of this outside point
                
                # Get all projected points for this outside point
                point_projected = projected_points_np[i]  # (n_vertices, 2)
                
                # Compute distances to all projected points
                distances_to_projected = np.linalg.norm(point_projected - point, axis=1)
                
                # Also compute distances to vertices
                distances_to_vertices = np.linalg.norm(hull_np - point, axis=1)
                
                # Find which is closer: closest vertex or closest projected point
                min_projected_dist = np.min(distances_to_projected)
                min_vertex_dist = np.min(distances_to_vertices)
                
                if min_projected_dist <= min_vertex_dist:
                    # Use projected point
                    closest_idx = np.argmin(distances_to_projected)
                    closest_point = point_projected[closest_idx]
                else:
                    # Use vertex
                    closest_idx = np.argmin(distances_to_vertices)
                    closest_point = hull_np[closest_idx]
                
                # Draw red line from point to closest point on hull
                ax.plot([point[0], closest_point[0]], [point[1], closest_point[1]], 
                       'r-', alpha=0.5, linewidth=1)
                
                # Add distance annotation
                ax.annotate(f'{min_distances_np[~is_inside_np][i]:.3f}', 
                          ((point[0] + closest_point[0])/2, (point[1] + closest_point[1])/2),
                          fontsize=8, color='red')
        elif len(outside_points) > 0:
            # Fallback: use nearest vertex if projected_points not provided
            for i, point in enumerate(outside_points):
                distances_to_vertices = np.linalg.norm(hull_np - point, axis=1)
                closest_vertex_idx = np.argmin(distances_to_vertices)
                closest_vertex = hull_np[closest_vertex_idx]
                
                # Draw red line from point to closest vertex
                ax.plot([point[0], closest_vertex[0]], [point[1], closest_vertex[1]], 
                       'r-', alpha=0.5, linewidth=1)
                
                # Add distance annotation
                ax.annotate(f'{min_distances_np[~is_inside_np][i]:.3f}', 
                          ((point[0] + closest_vertex[0])/2, (point[1] + closest_vertex[1])/2),
                          fontsize=8, color='red')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Distance Visualization (Red lines show min distances to hull)')
    
    def compute_separation_loss_vectorized(self, k: int = 2000) -> torch.Tensor:
        """
        Vectorized computation of separation loss.
        
        Args:
            k: number of closest points to check for each cluster pair
            
        Returns:
            Total separation loss
        """
        # Get transformed clusters and hulls
        transformed_clusters = self.apply_transformations_vectorized()  # (n_clusters, max_cluster_size, 2)
        transformed_hulls = self.apply_transformations_vectorized_hulls()  # (n_clusters, max_hull_vertices, 2)
        transformed_medoids = self.stacked_medoids + self.translations  # (n_clusters, 2)
        
        # Find k closest points for each cluster to each other cluster's medoid
        closest_indices = self.find_k_closest_points_vectorized(
            transformed_clusters, transformed_medoids, k
        )  # (n_clusters, n_clusters, k)
        
        total_violation = torch.tensor(0.0)
        
        # Process each cluster pair
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                if i != j:
                    # Get k closest points from cluster i to cluster j's medoid
                    point_indices = closest_indices[i, j]  # (k,)
                    
                    # Get the actual points
                    points_to_check = transformed_clusters[i][point_indices]  # (k, 2)
                    
                    # Check if these points are in cluster j's hull
                    hull_vertices_j = transformed_hulls[j]  # (max_hull_vertices, 2)
                    hull_mask_j = self.hull_masks[j]  # (max_hull_vertices,)
                    
                    violations = self.point_in_convex_hull_vectorized_fast(
                        points_to_check, hull_vertices_j, hull_mask_j, i, j
                    )  # (k,)
                    
                    total_violation = total_violation + violations.mean()
        
        return total_violation
    
    def compute_drift_loss(self) -> torch.Tensor:
        """
        Compute regularization loss to prevent clusters from drifting too far.
        Vectorized version.
        """
        # Vectorized computation of drift loss
        translation_penalties = torch.sum(self.translations ** 2, dim=1)  # (n_clusters,)
        drift_loss = self.translation_penalty_factor * translation_penalties.sum()

        if self.rotation_flag:
            rotation_penalties = self.rotation_angles ** 2  # (n_clusters,)
            drift_loss += self.rotation_penalty_factor * rotation_penalties.sum()
        
        return drift_loss
    
    def forward(self) -> torch.Tensor:
        """
        Compute total loss combining separation and drift penalties.
        """
        separation_loss = self.compute_separation_loss_vectorized()
        drift_loss = self.compute_drift_loss()
        
        total_loss = self.separation_weight * separation_loss + self.drift_weight * drift_loss
        return total_loss
    
    def apply_transformations(self) -> List[torch.Tensor]:
        """
        Apply rotations and translations to get current cluster positions.
        Returns list format for compatibility.
        """
        transformed_padded = self.apply_transformations_vectorized()
        
        # Convert back to list format
        transformed_clusters = []
        for i in range(self.n_clusters):
            size = self.cluster_sizes[i]
            transformed_clusters.append(transformed_padded[i, :size])
        
        return transformed_clusters
    
    def get_current_medoids(self) -> List[torch.Tensor]:
        """Get current medoid positions after transformations."""
        # Use the same transformation logic as in the loss function
        transformed_medoids = self.stacked_medoids + self.translations
        return [transformed_medoids[i] for i in range(self.n_clusters)]
    
    def test_distance_visualization(self, cluster_idx: int = 0, num_test_points: int = 20):
        """
        Test the distance visualization by creating a plot showing minimum distances to a cluster's hull.
        
        Args:
            cluster_idx: Index of the cluster to test
            num_test_points: Number of test points to generate
        """
        import matplotlib.pyplot as plt
        
        # Get the current state of the cluster
        transformed_clusters = self.apply_transformations()
        transformed_hulls = self.apply_transformations_vectorized_hulls()
        
        cluster_points = transformed_clusters[cluster_idx]
        hull_vertices = transformed_hulls[cluster_idx]
        hull_mask = self.hull_masks[cluster_idx]
        
        # Create test points around the cluster
        cluster_np = cluster_points.detach().numpy()
        hull_np = hull_vertices[hull_mask].detach().numpy()
        
        # Generate test points in a grid around the cluster
        x_min, x_max = cluster_np[:, 0].min() - 2, cluster_np[:, 0].max() + 2
        y_min, y_max = cluster_np[:, 1].min() - 2, cluster_np[:, 1].max() + 2
        
        x_coords = np.linspace(x_min, x_max, int(np.sqrt(num_test_points)))
        y_coords = np.linspace(y_min, y_max, int(np.sqrt(num_test_points)))
        
        test_points = []
        for x in x_coords:
            for y in y_coords:
                test_points.append([x, y])
        
        test_points = torch.tensor(test_points, dtype=torch.float32)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Call the function with visualization enabled
        violations = self.point_in_convex_hull_vectorized_fast(
            test_points, hull_vertices, hull_mask, 0, 0, 
            visualize_distances=True, ax=ax
        )
        
        plt.tight_layout()
        plt.show()
        
        print(f"Test completed. Generated {len(test_points)} test points.")
        print(f"Violations shape: {violations.shape}")
        print(f"Violations range: [{violations.min().item():.6f}, {violations.max().item():.6f}]")

def optimize_cluster_separation(clusters: List[np.ndarray], 
                              medoids: List[np.ndarray],
                              medoid_distances: np.ndarray,
                              max_iterations: int = 300,
                              separation_weight: float = 2.0,
                              drift_weight: float = 0.0,
                              translation_penalty_factor: float = 1.0,
                              rotation_penalty_factor: float = 1.0,
                              rotation_flag: bool = False,
                              rel_tol: float = 0.0005,
                              abs_tol: float = 0.001,
                              verbose: bool = True,
                              visualize: bool = True,
                              create_movie: bool = False,
                              movie_fps: float = 0.5,
                              frame_interval: int = 1,
                              movie_filename: str = "optimization_movie.gif",
                              optimizer_type: str = "sgd",
                              labels = None,
                              lr_start: float = 0.1,
                              lr_end: float = 1.0,
                              lr_warmup_epochs: int = 20) -> dict:
    """
    Optimize cluster positions to achieve separation.
    Args:
        clusters: List of cluster point arrays
        medoids: List of medoid positions
        medoid_distances: Distance matrix between medoids
        max_iterations: Maximum number of optimization iterations
        separation_weight: Weight for separation loss
        drift_weight: Weight for drift regularization
        tolerance: Convergence tolerance
        verbose: Whether to print progress
        visualize: Whether to show final visualization
        create_movie: Whether to create optimization movie
        movie_fps: Frames per second for the movie
        frame_interval: Capture frame every N iterations
        movie_filename: Output filename for the movie
        optimizer_type: 'adam' or 'sgd' (default)
        labels: Optional labels for visualization
        lr_start: Starting learning rate for the schedule
        lr_end: Final learning rate for the schedule
        lr_warmup_epochs: Number of epochs to warm up the learning rate
    """
    
    optimizer_model = ClusterSeparationOptimizer(clusters, medoids, medoid_distances, rotation_flag=rotation_flag, separation_weight=separation_weight, drift_weight=drift_weight, translation_penalty_factor=translation_penalty_factor, rotation_penalty_factor=rotation_penalty_factor)

    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(optimizer_model.parameters(), lr=lr_start)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(optimizer_model.parameters(), lr=lr_start)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
    
    losses = []
    frames = []
    medoid_paths = []  # Track medoid positions during optimization
    
    if verbose:
        print(f"Optimizing separation for {len(clusters)} clusters...")
        print(f"Total parameters: {sum(p.numel() for p in optimizer_model.parameters())}")
        if create_movie:
            print(f"Creating movie: {movie_filename} (fps: {movie_fps}, frame interval: {frame_interval})")
    
    for iteration in range(max_iterations):
        # Track medoid positions
        current_medoids = optimizer_model.get_current_medoids()
        medoid_positions = torch.stack(current_medoids).detach().cpu().numpy()
        medoid_paths.append(medoid_positions)

        # Learning rate schedule: start at lr_start, end at lr_end over lr_warmup_epochs
        if iteration < lr_warmup_epochs:
            current_lr = lr_start + (lr_end - lr_start) * (iteration / lr_warmup_epochs)
        else:
            current_lr = lr_end
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad()
        loss = optimizer_model()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        
        # Capture frame for movie if enabled
        if create_movie and iteration % frame_interval == 0:
            frame = create_optimization_frame(
                optimizer_model, iteration, loss.item(),
                title="Cluster Separation Optimization",
                medoid_paths=np.array(medoid_paths) if medoid_paths else None
            )
            frames.append(frame)
        
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item():.6f}")
        
        # Check convergence
        if iteration > 10 and (abs(losses[-10] - losses[-1]) < rel_tol or losses[-1] < abs_tol):
            if verbose:
                print(f"Converged at iteration {iteration}")
            break
    
    # Store final medoid positions
    current_medoids = optimizer_model.get_current_medoids()
    medoid_positions = torch.stack(current_medoids).detach().cpu().numpy()
    medoid_paths.append(medoid_positions)

    # Store medoid paths in the optimizer model for later visualization
    optimizer_model.medoid_paths = torch.tensor(np.array(medoid_paths), dtype=torch.float32)
    
    # Create movie if requested
    if create_movie and frames:
        if verbose:
            print(f"Saving movie with {len(frames)} frames...")
        
        # Save as GIF
        if len(frames) > 1:
            frames[0].save(
                movie_filename,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // movie_fps,  # Duration in milliseconds
                loop=0
            )
            if verbose:
                print(f"Movie saved as: {movie_filename}")
        else:
            # If only one frame, save as static image
            frames[0].save(movie_filename.replace('.gif', '.png'))
            if verbose:
                print(f"Single frame saved as: {movie_filename.replace('.gif', '.png')}")
    
    if visualize:
        visualize_optimization_with_labels(optimizer_model, medoid_paths=np.array(medoid_paths), labels=labels)

    # Create a results dictionary to store all the data
    results = {
        'optimizer_model': optimizer_model,
        'medoid_paths': np.array(medoid_paths),
        'labels': labels,
        'losses': losses
    }

    return results


def create_optimization_frame(optimizer_model: ClusterSeparationOptimizer, 
                            iteration: int, loss: float,
                            title: str = "Cluster Separation Optimization", 
                            point = None, p12 = None, medoid_paths = None) -> Image.Image:
    """
    Create a single frame for the optimization movie.
    
    Args:
        optimizer_model: The optimizer model with current state
        iteration: Current iteration number
        loss: Current loss value
        title: Title for the plot
        point: Optional point to highlight
        p12: Optional pair of points to highlight
        medoid_paths: Optional list of medoid positions for visualization
        
    Returns:
        PIL Image object of the frame
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, optimizer_model.n_clusters))
    
    # Plot original clusters
    ax1.set_title("Original Clusters")
    for i, (cluster, medoid) in enumerate(zip(optimizer_model.original_clusters, 
                                            optimizer_model.original_medoids)):
        cluster_np = cluster.detach().numpy()
        medoid_np = medoid.detach().numpy()
        
        ax1.scatter(cluster_np[:, 0], cluster_np[:, 1], 
                   c=[colors[i]], alpha=0.6, s=50, label=f'Cluster {i+1}')
        ax1.scatter(medoid_np[0], medoid_np[1], 
                   c=[colors[i]], s=200, marker='*', edgecolors='black')
        
        # Draw original convex hull
        if len(cluster_np) >= 3:
            try:
                hull = ConvexHull(cluster_np)
                hull_points = cluster_np[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])
                ax1.plot(hull_points[:, 0], hull_points[:, 1], 
                        color=colors[i], alpha=0.8, linewidth=2)
            except:
                pass
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot current clusters
    ax2.set_title(f"Current State (Iteration {iteration}, Loss: {loss:.6f})")
    transformed_clusters = optimizer_model.apply_transformations()
    current_medoids = optimizer_model.get_current_medoids()
    
    # Plot medoid paths if available
    if medoid_paths is not None and len(medoid_paths) > 0:
        for i in range(optimizer_model.n_clusters):
            # Plot the complete path up to the current iteration in background
            path_so_far = medoid_paths[:iteration+1, i, :]
            if len(path_so_far) > 1:
                ax2.plot(path_so_far[:, 0], path_so_far[:, 1], 
                         c=colors[i], alpha=0.3, linestyle='--', linewidth=1)
                # Add small black dots at each point in the path
                ax2.scatter(path_so_far[:, 0], path_so_far[:, 1], 
                           c=colors[i], s=10, alpha=0.6, zorder=1, edgecolors='black', linewidth=0.5)
    
    for i, (cluster, medoid) in enumerate(zip(transformed_clusters, current_medoids)):
        cluster_np = cluster.detach().numpy()
        medoid_np = medoid.detach().numpy()
        
        ax2.scatter(cluster_np[:, 0], cluster_np[:, 1], 
                   c=[colors[i]], alpha=0.6, s=50, label=f'Cluster {i+1}')
        ax2.scatter(medoid_np[0], medoid_np[1], 
                   c=[colors[i]], s=200, marker='*', edgecolors='black')
        
        # Draw current convex hull
        if len(cluster_np) >= 3:
            try:
                hull = ConvexHull(cluster_np)
                hull_points = cluster_np[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])
                ax2.plot(hull_points[:, 0], hull_points[:, 1], 
                        color=colors[i], alpha=0.8, linewidth=2)
                ax2.fill(hull_points[:, 0], hull_points[:, 1], 
                        color=colors[i], alpha=0.1)
            except:
                pass
    
    if point is not None:
        plotPoint = point.detach().numpy()
        ax2.scatter(plotPoint[0], plotPoint[1], s=50, c="blue")
    if p12 is not None:
        p10 = p12[0]
        p20 = p12[1]
        p10 = p10.detach().numpy()
        p20 = p20.detach().numpy()
        ax2.scatter(p10[0], p10[1], s=50, c="red") 
        ax2.scatter(p20[0], p20[1], s=50, c="red")
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.suptitle(f"{title} - Iteration {iteration}")
    plt.tight_layout()
    
    # Convert matplotlib figure to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # Close the figure to free memory
    
    return img


def create_optimization_movie(optimizer_model: ClusterSeparationOptimizer,
                            iterations: List[int],
                            losses: List[float],
                            movie_filename: str = "optimization_movie.gif",
                            movie_fps: int = 5,
                            title: str = "Cluster Separation Optimization",
                            point = None, p12 = None, medoid_paths = None) -> None:
    """
    Create a movie from a list of optimization states.
    
    Args:
        optimizer_model: The optimizer model (will be modified to show different states)
        iterations: List of iteration numbers
        losses: List of loss values corresponding to iterations
        movie_filename: Output filename for the movie
        movie_fps: Frames per second for the movie
        title: Title for the movie
        point: Optional point to highlight
        p12: Optional pair of points to highlight
    """
    frames = []
    
    print(f"Creating movie with {len(iterations)} frames...")
    
    for i, (iteration, loss) in enumerate(zip(iterations, losses)):
        # Update the model to the state at this iteration
        # Note: This is a simplified version - in practice you'd need to store
        # the model parameters at each iteration during optimization
        
        frame = create_optimization_frame(
            optimizer_model, iteration, loss, title, point, p12, medoid_paths
        )
        frames.append(frame)
        
        if i % 10 == 0:
            print(f"Created frame {i+1}/{len(iterations)}")
    
    # Save as GIF
    if len(frames) > 1:
        frames[0].save(
            movie_filename,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // movie_fps,  # Duration in milliseconds
            loop=0
        )
        print(f"Movie saved as: {movie_filename}")
    else:
        # If only one frame, save as static image
        frames[0].save(movie_filename.replace('.gif', '.png'))
        print(f"Single frame saved as: {movie_filename.replace('.gif', '.png')}")


def visualize_optimization(optimizer_model: ClusterSeparationOptimizer,
                          title: str = "Cluster Separation Optimization", point = None, p12 = None, make_summary = True, medoid_paths = None):
    """
    Visualize the original and optimized cluster positions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, optimizer_model.n_clusters))
    
    # Plot original clusters
    ax1.set_title("Original Clusters")
    for i, (cluster, medoid) in enumerate(zip(optimizer_model.original_clusters, 
                                            optimizer_model.original_medoids)):
        cluster_np = cluster.detach().numpy()
        medoid_np = medoid.detach().numpy()
        
        ax1.scatter(cluster_np[:, 0], cluster_np[:, 1], 
                   c=[colors[i]], alpha=0.6, s=50, label=f'Cluster {i+1}')
        ax1.scatter(medoid_np[0], medoid_np[1], 
                   c=[colors[i]], s=200, marker='*', edgecolors='black')
        
        # Draw original convex hull
        if len(cluster_np) >= 3:
            try:
                hull = ConvexHull(cluster_np)
                hull_points = cluster_np[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])
                ax1.plot(hull_points[:, 0], hull_points[:, 1], 
                        color=colors[i], alpha=0.8, linewidth=2)
            except:
                pass
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot optimized clusters
    ax2.set_title("Optimized Clusters")
    transformed_clusters = optimizer_model.apply_transformations()
    current_medoids = optimizer_model.get_current_medoids()
    
    if medoid_paths is not None:
        # Plot medoid paths in background first
        for i in range(optimizer_model.n_clusters):
            ax2.plot(medoid_paths[:, i, 0], medoid_paths[:, i, 1], 
                     c=colors[i], alpha=0.6, linestyle='--', linewidth=1, zorder=10)
            # Add small black dots at each point in the path
            ax2.scatter(medoid_paths[0, i, 0], medoid_paths[0, i, 1], marker='o', 
                       c=colors[i], s=50, alpha=0.8, zorder=10, edgecolors='black', linewidth=0.5)

    for i, (cluster, medoid) in enumerate(zip(transformed_clusters, current_medoids)):
        cluster_np = cluster.detach().numpy()
        medoid_np = medoid.detach().numpy()
        
        # ax2.scatter(cluster_np[:, 0], cluster_np[:, 1], 
        #            c=[colors[i]], alpha=0.6, s=100, label=f'Cluster {i+1}')
        ax2.scatter(medoid_np[0], medoid_np[1], 
                   c=[colors[i]], alpha=0.8, s=150, marker='*', edgecolors='black')
        
        # Draw optimized convex hull
        if len(cluster_np) >= 3:
            try:
                hull = ConvexHull(cluster_np)
                hull_points = cluster_np[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])
                ax2.plot(hull_points[:, 0], hull_points[:, 1], 
                        color=colors[i], alpha=0.8, linewidth=2)
                ax2.fill(hull_points[:, 0], hull_points[:, 1], 
                        color=colors[i], alpha=0.1)
            except:
                pass
    
    if point is not None:
        plotPoint = point.detach().numpy()
        ax2.scatter(plotPoint[0], plotPoint[1], s=50, c="blue")
    if p12 is not None:
        p10 = p12[0]
        p20 = p12[1]
        p10 = p10.detach().numpy()
        p20 = p20.detach().numpy()
        ax2.scatter(p10[0], p10[1], s=50, c="red") 
        ax2.scatter(p20[0], p20[1], s=50, c="red")
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    plt.show()

def visualize_optimization_with_labels(optimizer_model: ClusterSeparationOptimizer,
                          title: str = "Cluster Separation Optimization", point = None, p12 = None, make_summary = True, medoid_paths = None, labels = None):
    """
    Visualize the original and optimized cluster positions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot optimized cluster points with labels
    ax1.set_title("Cluster Points")
    transformed_clusters = optimizer_model.apply_transformations()
    current_medoids = optimizer_model.get_current_medoids()
    
    # Plot cluster points
    ax1.set_title("Cluster points")
    colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, optimizer_model.n_clusters))

    if labels is not None:
        # Convert labels (integers) to colors using jet colormap
        label_colors = []
        for label_array in labels:
            # Convert each integer label to its corresponding color from jet colormap
            color_array = [colors[label] for label in label_array]
            label_colors.append(color_array)
    else:
        label_colors = colors
    
    for i, (cluster, medoid) in enumerate(zip(transformed_clusters, current_medoids)):
        cluster_np = cluster.detach().numpy()
        medoid_np = medoid.detach().numpy()
        
        if labels is not None:
            ax1.scatter(cluster_np[:, 0], cluster_np[:, 1], 
                   c=label_colors[i], alpha=0.6, s=5, edgecolors='none')
        else:
            ax1.scatter(cluster_np[:, 0], cluster_np[:, 1], 
                   c=[label_colors[i]], alpha=0.6, s=5, edgecolors='none')

        # # Draw original convex hull
        # if len(cluster_np) >= 3:
        #     try:
        #         hull = ConvexHull(cluster_np)
        #         hull_points = cluster_np[hull.vertices]
        #         hull_points = np.vstack([hull_points, hull_points[0]])
        #         ax1.plot(hull_points[:, 0], hull_points[:, 1], 
        #                 color=colors[i], alpha=0.8, linewidth=2)
        #     except:
        #         pass
    
    # Calculate majority label color for each cluster
    cluster_majority_colors = []
    for i in range(len(transformed_clusters)):
        if labels is not None:
            cluster_labels = labels[i]
            unique, counts = np.unique(cluster_labels, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            # Map majority label to jet colormap
            cluster_majority_colors.append(colors[majority_label])
        else:
            cluster_majority_colors.append(colors[i])

    # Create legend for unique labels
    from matplotlib.lines import Line2D
    legend_elements = []
    
    if labels is not None:
        # Create legend for unique labels using the jet colormap
        all_labels = []
        for label_array in labels:
            all_labels.extend(label_array)
        unique_labels = np.unique(all_labels)
        for label in unique_labels:
            color = tuple(colors[label])
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        label=f'{label}'))
    else:
        # Fallback for when no labels are provided
        for i, color in enumerate(colors):
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        label=f'{i+1}'))
        
    ax1.legend(handles=legend_elements)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot optimized clusters
    ax2.set_title("Optimized Clusters")
    
    if medoid_paths is not None:
        # Plot medoid paths in background first
        for i in range(optimizer_model.n_clusters):
            ax2.plot(medoid_paths[:, i, 0], medoid_paths[:, i, 1], 
                     c=cluster_majority_colors[i], alpha=0.6, linestyle='--', linewidth=1, zorder=10)
            # Add small black dots at each point in the path
            ax2.scatter(medoid_paths[0, i, 0], medoid_paths[0, i, 1], marker='o', 
                       c=cluster_majority_colors[i], s=50, alpha=0.8, zorder=10, edgecolors='black', linewidth=0.5)

    for i, (cluster, medoid) in enumerate(zip(transformed_clusters, current_medoids)):
        cluster_np = cluster.detach().numpy()
        medoid_np = medoid.detach().numpy()
        
        # ax2.scatter(cluster_np[:, 0], cluster_np[:, 1], 
        #            c=[colors[i]], alpha=0.6, s=100, label=f'Cluster {i+1}')
        ax2.scatter(medoid_np[0], medoid_np[1], 
                   c=cluster_majority_colors[i], alpha=0.8, s=150, marker='*', edgecolors='black')
        
        # Draw optimized convex hull
        if len(cluster_np) >= 3:
            try:
                hull = ConvexHull(cluster_np)
                hull_points = cluster_np[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])
                ax2.plot(hull_points[:, 0], hull_points[:, 1], 
                        color=cluster_majority_colors[i], alpha=0.8, linewidth=2)
                ax2.fill(hull_points[:, 0], hull_points[:, 1], 
                        color=cluster_majority_colors[i], alpha=0.1)
            except:
                pass
    
    if point is not None:
        plotPoint = point.detach().numpy()
        ax2.scatter(plotPoint[0], plotPoint[1], s=50, c="blue")
    if p12 is not None:
        p10 = p12[0]
        p20 = p12[1]
        p10 = p10.detach().numpy()
        p20 = p20.detach().numpy()
        ax2.scatter(p10[0], p10[1], s=50, c="red") 
        ax2.scatter(p20[0], p20[1], s=50, c="red")
    
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    plt.show()


# Example usage
def main():
    """Demonstrate the optimized cluster separation optimization."""
    
    # Create overlapping clusters
    np.random.seed(42)
    
    # Cluster 1: centered around (2, 2)
    cluster1 = np.random.randn(15, 2) * 0.8 + np.array([2, 2])
    medoid1 = np.array([2, 2])
    
    # Cluster 2: centered around (3, 1) - overlapping with cluster 1
    cluster2 = np.random.randn(12, 2) * 0.6 + np.array([3, 1])
    medoid2 = np.array([3, 1])
    
    # Cluster 3: centered around (1, 3) - also overlapping
    cluster3 = np.random.randn(10, 2) * 0.5 + np.array([1, 3])
    medoid3 = np.array([1, 3])
    
    clusters = [cluster1, cluster2, cluster3]
    medoids = [medoid1, medoid2, medoid3]
    
    # Create medoid distances matrix (simple Euclidean distances)
    medoid_distances = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                medoid_distances[i, j] = np.linalg.norm(medoids[i] - medoids[j])
    
    print("Initial cluster setup:")
    for i, (cluster, medoid) in enumerate(zip(clusters, medoids)):
        print(f"Cluster {i+1}: {len(cluster)} points, medoid at {medoid}")
    
    # Test optimized version
    print("\n" + "="*60)
    print("Testing optimized vectorized optimization...")
    
    import time
    start_time = time.time()
    
    results = optimize_cluster_separation(
        clusters=clusters,
        medoids=medoids,
        medoid_distances=medoid_distances,
        max_iterations=100,
        lr_start=0.02,
        separation_weight=2.0,
        drift_weight=0.1,
        verbose=True,
        create_movie=True,  # Enable movie creation
        movie_fps=3,        # 3 frames per second
        frame_interval=10,  # Capture frame every 10 iterations
        movie_filename="cluster_optimization.gif"
    )
    
    end_time = time.time()
    print(f"Optimized vectorized optimization completed in {end_time - start_time:.2f} seconds")
    
    # Extract the optimizer model from results
    optimized_model = results['optimizer_model']
    
    # Test distance visualization
    print("\n" + "="*60)
    print("Testing distance visualization...")
    optimized_model.test_distance_visualization(cluster_idx=0, num_test_points=25)
    
    # Visualize results
    visualize_optimization(optimized_model, "Optimized 3-Cluster Separation")


if __name__ == "__main__":
    main()