"""
A O(n^2) thompson metric evaluation using inexact linear solvers

References
---------
"""

import numpy as np
from gradient_descent import linear_inexact_solve
import generate_random_spd

def gen_eig(A,B):
    """
    Calculate the general eigenvalues of A and B
    """
    # Assume A and B are symetric
    dimension = A.shape[0]
    print(f"dimension: {dimension}")
    init_vector = np.random.uniform(low=0.0,
                                    high=1.0,
                                    size=dimension)

    init_vector = init_vector/B_norm(init_vector,B)
    w = init_vector
    for i in range(1000):
        beta = B_norm(w,A)/B_norm(w,B)
        w = linear_inexact_solve(A,B,w,beta)
        w = w/B_norm(w,B)
        if i%100 == 0:
            print(beta)
    return w, beta



def B_norm(u,B):
    """Calculate the B norm of a vector"""
    return np.linalg.norm(u.T@B@u)


if __name__ == "__main__":
    i = 3
    A = generate_random_spd.single_spd_gen(int((i+3)*10), float(36.0))
    B = generate_random_spd.single_spd_gen(int((i+3)*10), float(36.0))
    print(gen_eig(A,B))
