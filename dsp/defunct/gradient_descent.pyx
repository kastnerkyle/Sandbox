cimport numpy as np
import numpy as np
cimport cython
from cython cimport bool
@cython.boundscheck(False)
@cython.wraparound(False)
def gradient_descent(np.ndarray[double, ndim=2] A,
                     np.ndarray[double, ndim=2] U,
                     np.ndarray[double, ndim=2] V,
                     np.ndarray[long int, ndim=2] Z,
                     long int K,
                     long int R,
                     long int N,
                     long int M,
                     double l,
                     double b):
    cdef unsigned int r,i,j,k
    cdef double e
    for i in range(N):
        for i in range(N):
            for j in range(M):
                if Z[i,j] > 0:
                    #for k in range(K):
                    #    e = A[i,j]-U[i,k]*V[k,j]
                    #    U[i,k] = U[i,k] + l*(e*V[k,j]-b*U[i,k])
                    #    V[k,j] = V[k,j] + l*(e*U[i,k]-b*V[k,j])
                    e = A[i,j]-np.dot(U[i,:],V[:,j]) 
                    U[i,:] = U[i,:] + l*(e*V[:,j]-b*U[i,:])
                    V[:,j] = V[:,j] + l*(e*U[i,:]-b*V[:,j])
    return np.dot(U,V)
