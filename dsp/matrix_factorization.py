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
    e = 0
    for i in xrange(K):
        A_ += S[i]*np.outer(U.T[i],VT[i])
        e += (A[i,:]-A_[i,:])**2
    RMSE = e
    return A_, RMSE

def PMF(input_matrix, approx=50, iterations=30, learning_rate=.001, regularization_rate=.1, randomize=True, print_status=True):
    A = input_matrix
    K = approx
    itr = iterations
    a = learning_rate
    b = regularization_rate
    iterations = itr = 30
    learning_rate = a = .001
    regularization_rate = b = .1
    N = len(A)
    M = len(A[0])
    U = np.random.randn(N,K)
    V = np.random.randn(K,M)
    if randomize:
        import random
    RMSE=[]
    if print_status:
        print "Starting PMF"
    for r in xrange(itr):
        e = 0
        r1 = range(len(A))
        random.shuffle(r1) if randomize else True
        for i in r1:
            r2 = range(len(A[i]))
            random.shuffle(r2) if randomize else True
            for j in r2:
                if A[i][j] > 0:
                    eij = A[i][j] - np.dot(U[i,:],V[:,j])
                    e += eij**2
                    for k in xrange(K):
                        U[i][k] = U[i][k] + a * (eij * V[k][j] - b * U[i][k])
                        V[k][j] = V[k][j] + a * (eij * U[i][k] - b * V[k][j])
        RMSE.append(e)
        if print_status:
            print "Iteration " + `r` + ": RMSE " + `e`
    A_ = np.dot(U,V)
    return A_,RMSE

def constrained_PMF(input_matrix, approx=50, iterations=30, learning_rate=.001, regularization_rate=.1, randomize=True, print_status=True):
    A = input_matrix
    K = approx
    itr = iterations
    a = learning_rate
    b = regularization_rate
    N = len(A)
    M = len(A[0])
    Y = np.random.randn(N,K)
    W = np.random.randn(N,K)
    V = np.random.randn(K,M)
    if randomize:
        import random
    RMSE = []
    if print_status:
        print "Starting constrained PMF"
    for r in xrange(itr):
        e = 0
        r1 = range(N)
        random.shuffle(r1) if randomize else True
        for i in r1:
            r2 = range(M)
            random.shuffle(r2) if randomize else True
            for j in r2:
                if A[i][j] > 0:
                    num = 0
                    denom = 1
                    for k in xrange(K):
                        if A[i][k] > 0:
                            num += W[i][k]
                            denom += 1.
                    s = num/denom
                    eij = (A[i][j] - np.dot(Y[i,:] + s,V[:,j]))
                    e += eij**2
                    for k in xrange(K):
                        Y[i][k] = Y[i][k] + a * (eij*V[k,j] + b*Y[i][k])
                        W[i][k] = W[i][k] + a * (eij*V[k,j] + b*W[i][k])
                        V[k][j] = V[k][j] + a * (eij*(Y[i,k]+s) + b*V[k][j])
        RMSE.append(e)
        if print_status:
            print "Iteration " + `r` + ": RMSE " + `e`
    A_ = np.dot(Y+W,V)
    return A_,RMSE

#Rework of lena example
#From https://gist.github.com/thearn/5424219
#Full matrix SVD, low rank approximation
approx = K = 50
A = lena()
A_,RMSE=lowrank_SVD(A,K)
plot.figure()
plot.imshow(A, cmap=cm.gray)
plot.figure()
plot.imshow(A_, cmap=cm.gray)

#Sparse matrix setup
sparseness = .95
A = lena()
for i in xrange(len(A)):
    for j in xrange(len(A[i])):
        if np.random.rand() < sparseness:
            A[i][j] = 0.

#Sparse matrix, regular SVD example, low rank approximation
A_,RMSE=lowrank_SVD(A,K)
plot.figure()
plot.imshow(A, cmap=cm.gray)
plot.figure()
plot.imshow(A_, cmap=cm.gray)

#Sparse matrix, gradient descent example
A_,RMSE=PMF(A,K)
plot.figure()
plot.imshow(A, cmap=cm.gray)
plot.figure()
plot.imshow(A_, cmap=cm.gray)

#Sparse matrix, constrained gradient descent example
A_,RMSE=constrained_PMF(A,K)
plot.figure()
plot.imshow(A, cmap=cm.gray)
plot.figure()
plot.imshow(A_, cmap=cm.gray)
plot.show()
