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
    Z = np.asarray(A > 0,dtype=np.int)
    A1d = np.ravel(A)
    mean = np.mean(A1d)
    A = A-mean
    K = approx
    R = itr = iterations
    l = learning_rate
    b = regularization_rate
    N = A.shape[0]
    M = A.shape[1]
    U = np.random.randn(N,K)
    V = np.random.randn(K,M)
    for r in range(R):
        for i in range(N):
            for j in range(M):
                if Z[i,j] > 0:
                    e = A[i,j] - np.dot(U[i,:],V[:,j])
                    U[i,:] = U[i,:] + l*(e*V[:,j] - b*U[i,:])
                    V[:,j] = V[:,j] + l*(e*U[i,:] - b*V[:,j])
    A_ = np.dot(U,V)
    return A_+mean

#Rework of lena example
#From https://gist.github.com/thearn/5424219
#Full matrix SVD, low rank approximation
approx = K = 20
iterations = I = 10
A = np.asarray(lena(),dtype=np.double)
A_= lowrank_SVD(A,approx=K)
plot.figure()
plot.title("Low Rank SVD (full matrix) RMSE = ")
plot.imshow(A_, cmap=cm.gray)

#Sparse matrix setup
sparseness = .85
A = lena()
for i in xrange(A.shape[0]):
    for j in xrange(A.shape[1]):
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
