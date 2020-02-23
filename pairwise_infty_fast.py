"""
All image search ranking related functionalities
"""
from scipy.spatial.distance import cdist, pdist
import numpy as np
import time
# from numba import double
from numba import jit
# from numba.decorators import jit, autojit



# --------- Dummy Test variables to be inserted around line 36---------
total= 100
total_1 = 100
vector_num = 9
feature_vectors_1 = np.random.rand(total,60*vector_num)
feature_vectors_2 = np.random.rand(total_1,60*vector_num)
def norm(local_vectors_1, local_vectors_2):
    norm = cdist(local_vectors_1.reshape(vector_num,60), local_vectors_2.reshape(vector_num,60)).min()
    return norm
# Calculate pairwise distance between new entries and all entries(including the new ones)
# new_scores = pdist(feature_vectors, metric=norm)
start = time.time()
new_scores = cdist(feature_vectors_1, feature_vectors_2, norm)
print(time.time()-start)
# --------- Dummy Tests Setup End ---------
x = feature_vectors_1.reshape(total,vector_num,60)
y = feature_vectors_2.reshape(total,vector_num,60)
@jit(nopython=True)
def pairwise_numba(X, Y):
    M_X = X.shape[0]
    M_Y = Y.shape[0]
    vector_num = X.shape[1]
    N = X.shape[2]
    # D = np.empty((M_X, M_Y), dtype=np.float)
    D = np.empty((M_X, M_Y))
    for i in range(M_X):
        for j in range(M_Y):
            metric = 1000.0
            for k_x in range(vector_num):
                for k_y in range(vector_num):
                # for k_y in range(k_x+1):
                    d=0.0
                    for l in range(N):
                        d += (X[i,k_x,l]-Y[j,k_y,l])**2
                    if d < metric:
                        metric = d
            D[i, j] = np.sqrt(metric)
    return D
# dist = pairwise_python(x,y)
# print(dist.shape)
# print(time.time()-start)
# print(np.testing.assert_almost_equal(dist, new_scores))
# pairwise_numba = jit(pairwise_python)
start = time.time()
dist = pairwise_numba(x,y)
print(time.time()-start)
print(np.testing.assert_almost_equal(dist, new_scores))
