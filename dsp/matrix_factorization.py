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
    A = input_matrix
    K = approx
    itr = iterations
    a = learning_rate
    b = regularization_rate
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
        #Current failed attempt at matrix version
        #I = A > 0
        #for k in xrange(K):
        #    V[k,:] = V[k,:] + a*(np.dot(U[:,k],np.dot(I,np.dot(U[:,k],V[k,:])-A))-b*V[k,:])
        #    U[:,k] = U[:,k] + a*(np.dot(V[k,:],np.dot(I,np.dot(U[:,k],V[k,:])-A))-b*U[:,k])
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

def KPMF(input_matrix, approx=50, iterations=30, learning_rate=.001, regularization_rate=.1, randomize=True, print_status=True):
    A = input_matrix
    K = approx
    A = input_matrix
    K = approx
    itr = iterations
    a = learning_rate
    b = regularization_rate
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
        #Current failed attempt at matrix version
        #I = A > 0
        #for k in xrange(K):
        #    V[k,:] = V[k,:] + a*(np.dot(U[:,k],np.dot(I,np.dot(U[:,k],V[k,:])-A))-b*V[k,:])
        #    U[:,k] = U[:,k] + a*(np.dot(V[k,:],np.dot(I,np.dot(U[:,k],V[k,:])-A))-b*U[:,k])
        #e = 0
        e = A - np.dot(U,V)
        r1 = range(len(A))
        random.shuffle(r1) if randomize else True
        for i in r1:
            r2 = range(len(A[i]))
            random.shuffle(r2) if randomize else True
            for j in r2:
                if A[i][j] > 0:
                    #e += eij**2
                    for k in xrange(K):
                        U[i,k] = U[i,k] + a * (e[i,j] * V[k,j] - b * U[i,k])
                        V[k,j] = V[k,j] + a * (e[i,j] * U[i,k] - b * V[k,j])
        RMSE.append(np.sum(np.sum(e))**2)
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
                if A[i,j] > 0:
                    div = float(sum(A[i,:] > 0))
                    for k in xrange(K):
                        n = 0.
                        if A[i,k] > 0:
                            n += W[k,:]/div
                    eij = A[i][j] - np.dot(Y[i,:] + n,V[:,j])
                    e += eij**2
                    for k in xrange(K):
                        Y[i][k] = Y[i][k] + a * (eij*V[k,j] - b*Y[i][k])
                        W[i][k] = W[i][k] + a * (eij*V[k,j] - b*W[i][k])
                        V[k][j] = V[k][j] + a * (eij*(Y[i,k]+W[i,k]) - b*V[k][j])
        RMSE.append(e)
        if print_status:
            print "Iteration " + `r` + ": RMSE " + `e`
    A_ = np.dot(Y+W,V)
    return A_,RMSE

#Rework of lena example
#From https://gist.github.com/thearn/5424219
#Full matrix SVD, low rank approximation
approx = K = 50
iterations = I = 10
A = lena()
A_,RMSE=lowrank_SVD(A,approx=K)
plot.figure()
plot.title("Low Rank SVD (full matrix) RMSE = " + `RMSE[-1]`)
plot.imshow(A_, cmap=cm.gray)

#Sparse matrix setup
sparseness = .85
A = lena()
for i in xrange(len(A)):
    for j in xrange(len(A[i])):
        if np.random.rand() < sparseness:
            A[i][j] = 0.

#Sparse lena
plot.figure()
plot.title("Sparse Lena")
plot.imshow(A, cmap=cm.gray)

#Sparse matrix, regular SVD example, low rank approximation
A_,RMSE=lowrank_SVD(A,approx=K)
plot.figure()
plot.title("Low Rank SVD, RMSE = " + `RMSE[-1]`)
plot.imshow(A_, cmap=cm.gray)

#Sparse matrix, gradient descent example
A_,RMSE=PMF(A,approx=K,iterations=I)
plot.figure()
plot.title("PMF, RMSE = " + `RMSE[-1]`)
plot.imshow(A_, cmap=cm.gray)
plot.show()

#Sparse matrix, constrained gradient descent example
#A_,RMSE=constrained_PMF(A,approx=K,iterations=I)
#plot.figure()
#plot.title("Constrained PMF, RMSE = " + `RMSE[-1]`)
#plot.imshow(A_, cmap=cm.gray)
#plot.show()
