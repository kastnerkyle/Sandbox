#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.cm as cm
from scipy.misc import lena
#Example from
#Singular Value Decomposition Tutorial
#Kirk Baker
#pg 17

#A = np.asarray([[3,1,1],[-1, 3, 1]])
#U,S,VT = np.linalg.svd(A, full_matrices=False)
#S = np.diag(S)
#A_ = np.dot(U, np.dot(S, VT))
#print np.allclose(A, A_)

def lowrank_SVD(input_matrix, approx=50):
    U,S,VT = np.linalg.svd(A, full_matrices=False)
    A_ = np.zeros((len(U), len(VT)))
    for i in xrange(K):
        A_ += S[i]*np.outer(U.T[i],VT[i])
    return A_

def PMF(input_matrix, approx=50, iterations=30, learning_rate=.001, regularization_rate=.1):
    A = input_matrix
    K = approx
    R = itr = iterations
    l = learning_rate
    b = regularization_rate
    N = A.shape[0]
    M = A.shape[1]
    U = np.random.randn(N,K)
    V = np.random.randn(K,M)
    Z = np.asarray(A > 0,dtype=np.int)
    opt = False
    opt = True
    from scipy.weave import blitz
    if not opt:
        for r in range(itr):
            for i in range(N):
                for j in range(M):
                    if Z[i,j]:
                        e = A[i,j] - np.dot(U[i,:],V[:,j])
                        U[i,:] = U[i,:] + l*(e*V[:,j] - b*U[i,:])
                        V[:,j] = V[:,j] + l*(e*U[i,:] - b*V[:,j])
    else:
    #http://technicaldiscovery.blogspot.com/2011/06/speeding-up-python-numpy-cython-and.html
        from scipy.weave import inline
        from scipy.weave import converters
        weave_options = {'extra_compile_args': ['-O3'],
                         'compiler': 'gcc'}
        code = \
r"""
int r,i,j,k;
double e;
for(r=0; r<R; r++){
    for(i=0; i<N; i++){
        for(j=0; j<M; j++){
            for(k=0; k<K; k++){
                if(Z(i,j)){
                    e = A(i,j)-(U(i,k)*V(k,j));
                    U(i,k) += l*(e*V(k,j)-b*U(i,k));
                    V(k,j) += l*(e*U(i,k)-b*V(k,j));
                }
            }
        }
    }
}
"""
        inline(code,
               ['A','K','R','l','b','N','M','U','V','Z'],
               type_converters=converters.blitz,
               **weave_options)

    A_ = np.dot(U,V)
    return A_

#Rework of lena example
#From https://gist.github.com/thearn/5424219
#Full matrix SVD, low rank approximation
approx = K = 50
iterations = I = 10
A = lena()
A_=lowrank_SVD(A,approx=K)
plot.figure()
plot.title("Low Rank SVD (full matrix) RMSE = ")
plot.imshow(A_, cmap=cm.gray)

#Sparse matrix setup
sparseness = .85
A = lena()
for i in xrange(len(A)):
    for j in xrange(len(A[i])):
        if np.random.rand() < sparseness:
            A[i,j] = 0.

#Sparse lena
plot.figure()
plot.title("Sparse Lena")
plot.imshow(A, cmap=cm.gray)

#Sparse matrix, regular SVD example, low rank approximation
A_=lowrank_SVD(A,approx=K)
plot.figure()
plot.title("Low Rank SVD, RMSE = ")
plot.imshow(A_, cmap=cm.gray)

#Sparse matrix, gradient descent example
A_=PMF(A,approx=K,iterations=I)
plot.figure()
plot.title("PMF, RMSE = ")
plot.imshow(A_, cmap=cm.gray)
plot.show()

#Sparse matrix, constrained gradient descent example
#A_,RMSE=KPMF(A,approx=K,iterations=I)
#plot.figure()
#plot.title("Kernelized PMF, RMSE = " + `RMSE[-1]`)
#plot.imshow(A_, cmap=cm.gray)
#plot.show()
