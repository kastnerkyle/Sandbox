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

#Rework of lena example
#From https://gist.github.com/thearn/5424219
#Full matrix SVD, low rank approximation
approx = K = 50
A = lena()
U,S,VT = np.linalg.svd(A, full_matrices=False)
A_ = np.zeros((len(U), len(VT)))
for i in xrange(K):
    A_ += S[i]*np.outer(U.T[i],VT[i])
plot.imshow(A, cmap=cm.gray)
plot.figure()
plot.imshow(A_, cmap=cm.gray)
plot.show()

#Sparse matrix setup
sparseness = .5
A = lena()
for i in xrange(len(A)):
    for j in xrange(len(A[i])):
        if np.random.rand() < sparseness:
            A[i][j] = 0.

#Regular SVD example, low rank approximation
U,S,VT = np.linalg.svd(A, full_matrices=False)
A_ = np.zeros((len(U), len(VT)))
for i in xrange(K):
    A_ += S[i]*np.outer(U.T[i],VT[i])
plot.imshow(A, cmap=cm.gray)
plot.figure()
plot.imshow(A_, cmap=cm.gray)
plot.show()

#Sparse matrix, gradient descent example
iterations = itr = 10
learning_rate = a = .0002
regularization_rate = b = .02
N = len(A)
M = len(A[0])
P = np.random.randn(N,K)
Q = np.random.randn(K,M)
for r in xrange(itr):
    e = 0
    for i in xrange(len(A)):
        for j in xrange(len(A[i])):
            if A[i][j] > 0:
                eij = A[i][j] - np.dot(P[i,:],Q[:,j])
                e = e + (A[i][j] - np.dot(P[i,:],Q[:,j]))**2
                for k in xrange(K):
                    P[i][k] = P[i][k] + a * (2 * eij * Q[k][j] - b * P[i][k])
                    Q[k][j] = Q[k][j] + a * (2 * eij * P[i][k] - b * Q[k][j])
                    e = e + (b/2) * (P[i][k]**2 + Q[k][j]**2)
    if e < 0.001:
        break

A_ = np.dot(P,Q)
plot.imshow(A, cmap=cm.gray)
plot.figure()
plot.imshow(A_, cmap=cm.gray)
plot.show()
