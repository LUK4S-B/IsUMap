import os
import sys

# Set the path to the directory containing `isumap.py`
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_CURRENT = os.path.join(SCRIPT_DIR, "../src/")
scriptPath = os.path.abspath(PATH_CURRENT)
sys.path.append(scriptPath)

from isumap import isumap
from distance_graph_generation import distance_graph_generation

import numpy as np

# Create asymmetric matrix where M[i,j] != M[j,i]
np.random.seed(42)
n = 5
M = np.random.rand(n, n) * 10
np.fill_diagonal(M, 0)  # Distance from point to itself is 0

# Make it clearly asymmetric
M[0, 1] = 2.0
M[1, 0] = 8.0
M[2, 3] = 1.5
M[3, 2] = 6.5

print("Asymmetric Distance Matrix M:")
print("(M[i,j] = distance FROM point i TO point j)")
print(M.round(2))
print()

# Check if matrix is symmetric
is_symmetric = np.allclose(M, M.T)
print(f"Is matrix symmetric? {is_symmetric}")
print()

D = distance_graph_generation(M, 3, normalize = False, distBeyondNN = False, dataIsDistMatrix = True, directedDistances = True)

print(D.round(2))