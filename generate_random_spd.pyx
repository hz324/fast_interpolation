"""
Generate random Symetric Positive Definite matrices
for benchmarking the SPD interpolation algorithm
The generation process involves generating n +ve eigenvalues
with a set of orthogonal orthonormal vectors
via the householder transformation
"""

import numpy as np
cimport numpy as np

# Define numpy datatype for arrays
DTYPE = np.float

ctypedef np.float_t DTYPE_t

def single_spd_gen(int dimension, float max_eigenvalue):
    cdef np.ndarray eigenvalues = np.zeros(dimension, dtype=DTYPE)
    cdef np.ndarray vector_seed = np.zeros(dimension, dtype=DTYPE)
    cdef np.ndarray eigenvectors = np.zeros([dimension, dimension], dtype=DTYPE)
    cdef np.ndarray spd = np.zeros([dimension, dimension], dtype=DTYPE)

    eigenvalues = np.random.uniform(low=1.3e-38,
                                    high=max_eigenvalue,
                                    size=dimension)

    vector_seed = np.random.uniform(low=0.0,
                                    high=1.0,
                                    size=dimension)

    vector_seed = vector_seed/np.linalg.norm(vector_seed)
    eigenvectors = np.eye(dimension, dtype=DTYPE) - 2*vector_seed*vector_seed.T
    print(eigenvectors)
    spd = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)),eigenvectors.T)

    return spd
