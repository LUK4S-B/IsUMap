import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from time import time
from data_and_plots import printtime

def hellinger_distance_matrix(sparse_counts):
    """
    Compute the Hellinger distance matrix between words represented as sparse count vectors.
    
    The Hellinger distance between two probability distributions P and Q is defined as:
    H(P, Q) = (1/√2) * ||√P - √Q||₂
    
    Where ||·||₂ is the L2 (Euclidean) norm.
    
    For discrete distributions with counts c₁, c₂, ..., cₙ, we first normalize to probabilities:
    P(i) = c_i / Σⱼ c_j
    
    Then the Hellinger distance becomes:
    H(P, Q) = (1/√2) * √(Σᵢ (√P(i) - √Q(i))²)
    
    This can be simplified to:
    H(P, Q) = √(1 - Σᵢ √(P(i) * Q(i)))
    
    Parameters:
    -----------
    sparse_counts : scipy.sparse matrix of shape (n_words, n_features)
        Sparse matrix where rows are words and columns are count features
        
    Returns:
    --------
    distance_matrix : numpy.ndarray of shape (n_words, n_words)
        Symmetric matrix where entry (i,j) is the Hellinger distance between word i and word j
    """
    
    # Convert to dense format for efficient computation (assuming reasonable size)
    # If memory is a concern, see the alternative sparse implementation below
    if sp.issparse(sparse_counts):
        counts = sparse_counts.toarray()
    else:
        counts = sparse_counts
    
    # Step 1: Normalize each word's count vector to probability distribution
    # P(i) = c_i / Σⱼ c_j (normalize each row to sum to 1)
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero for empty rows (words with no counts)
    row_sums[row_sums == 0] = 1
    prob_matrix = counts / row_sums
    
    # Step 2: Take square root of probabilities
    # We need √P(i) for the Hellinger distance formula
    sqrt_prob_matrix = np.sqrt(prob_matrix)
    
    # Step 3: Compute pairwise dot products between √P vectors
    # This gives us Σᵢ √(P(i) * Q(i)) for each pair of words
    dot_products = sqrt_prob_matrix @ sqrt_prob_matrix.T
    
    # Step 4: Apply Hellinger distance formula
    # H(P, Q) = √(1 - Σᵢ √(P(i) * Q(i)))
    # Clip to avoid numerical issues with floating point precision
    hellinger_matrix = np.sqrt(np.clip(1.0 - dot_products, 0.0, 1.0))
    
    return hellinger_matrix

def hellinger_distance_matrix_vectorized_sparse(sparse_counts, verbose=False):
    """
    Alternative vectorized implementation using the dot product formula.
    This can be more efficient for some matrix sizes.
    
    Uses the identity: H(P,Q) = √(1 - Σᵢ √(P(i) * Q(i)))
    Which allows us to compute all pairwise distances via matrix multiplication.
    """
    
    t0 = time()
    if verbose:
        print("Start Hellinger distance computation")
    # Ensure CSR format
    if not sp.isspmatrix_csr(sparse_counts):
        sparse_counts = sparse_counts.tocsr()
    t1 = time()
    if verbose:
        printtime("sparsity ensured", t1-t0)
    t0 = time()
    
    # Normalize to probabilities (sparse operations)
    row_sums = np.array(sparse_counts.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    t1 = time()
    if verbose:
        printtime("row sums computed", t1-t0)
    t0 = time()
    
    # Scale rows to get probabilities
    row_sums_inv = sp.diags(1.0 / row_sums, format='csr')
    t1 = time()
    if verbose:
        printtime("row sums inv computed", t1-t0)
    t0 = time()
    sparse_probs = row_sums_inv @ sparse_counts
    t1 = time()
    if verbose:
        printtime("probabilities computed", t1-t0)
    t0 = time()
    
    # Take square root element-wise
    sparse_sqrt_probs = sparse_probs.copy()
    t1 = time()
    if verbose:
        printtime("copy computed", t1-t0)
    t0 = time()
    sparse_sqrt_probs.data = np.sqrt(sparse_sqrt_probs.data)
    t1 = time()
    if verbose:
        printtime("sqrt computed", t1-t0)
    t0 = time()
    
    # Compute dot products: √P @ √P.T gives Σᵢ √(P(i) * Q(i)) for all pairs
    # This is where we finally need dense computation
    sqrt_prob_dense = sparse_sqrt_probs.toarray()
    t1 = time()
    if verbose:
        printtime("to dense array computed", t1-t0)
    t0 = time()
    dot_products = sqrt_prob_dense @ sqrt_prob_dense.T
    t1 = time()
    if verbose:
        printtime("dot products computed", t1-t0)
    t0 = time()
    # Apply Hellinger formula: H(P,Q) = √(1 - dot_product)
    hellinger_matrix = np.sqrt(np.clip(1.0 - dot_products, 0.0, 1.0))
    t1 = time()
    if verbose:
        printtime("hellinger distance computed", t1-t0)
    
    return hellinger_matrix


