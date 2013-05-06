#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.cm as cm
from scipy.misc import lena
#To get the latest scipy (for banded matrix generation
#sudo apt-get install python python-dev gfortran libatlas-base-dev
#sudo pip install scipy
import scipy.sparse as sp
import scipy.linalg as sl
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

def KPMF(input_matrix, approx=50, iterations=30, learning_rate=.001, adjacency_width=5, adjacency_strength=.5):
    A = input_matrix
    Z = np.asarray(A > 0,dtype=np.int)
    A1d = np.ravel(A)
    mean = np.mean(A1d)
    A = A-mean
    K = approx
    R = itr = iterations
    l = learning_rate
    N = A.shape[0]
    M = A.shape[1]
    U = np.random.randn(N,K)
    V = np.random.randn(K,M)
    #Using diffusion kernel
    #U are the rows, we use an adjacency matrix CU
    #This matrix assumes that rows +- are connected to each other
    #V are the columns are connected as well
    #This forms a spatial smoothness graph
    #See Kernelized Probabilistic Matrix Factorization: Exploiting Graphs and Side Information
    #T. Zhou, H. Shan, A. Banerjee, G. Sapiro
    bw = adjacency_width
    #Use scipy.sparse.diags to generate band matrix with bandwidth = 2*adjacency_width
    #[1 1 1 1 0 0]
    #[0 1 1 1 1 0]
    #[0 0 1 1 1 1]
    CU = sp.diags([1]*(2*bw+1),range(-bw,bw+1),shape=(N,N)).todense()
    DU = np.diagflat(np.sum(CU,1))
    CV = sp.diags([1]*(2*bw+1),range(-bw,bw+1),shape=(M,M)).todense()
    DV = np.diagflat(np.sum(CV,1))
    LU = DU - CU
    LV = DV - CV
    beta = adjacency_strength
    KU = sl.expm(-beta*LU)
    KV = sl.expm(-beta*LV)
    SU = np.linalg.pinv(KU)
    SV = np.linalg.pinv(KV)
    for r in range(R):
        for i in range(N):
            for j in range(M):
                if Z[i,j] > 0:
                    e = A[i,j] - np.dot(U[i,:],V[:,j])
                    U[i,:] = U[i,:] + l*(e*V[:,j] - np.dot(SU[i,:],U))
                    V[:,j] = V[:,j] + l*(e*U[i,:] - np.dot(V,SV[:,j]))
    A_ = np.dot(U,V)
    return A_+mean

#Rework of lena example
#From https://gist.github.com/thearn/5424219
#Sparse lena
A = np.asarray(lena(),dtype=np.double)
plot.figure()
plot.title("Original Lena")
plot.imshow(A, cmap=cm.gray)

#Full matrix SVD, low rank approximation
approx = K = 30
iterations = I = 10
A_= lowrank_SVD(A,approx=K)
plot.figure()
plot.title("Low Rank SVD (full matrix) RMSE = ")
plot.imshow(A_, cmap=cm.gray)

#Make lena sparse matrix setup, sparseness is the percentage of deleted pixels
sparseness = .75
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
RMSE = np.sum(np.sum(A-A_))/float(A.shape[0]*A.shape[1])
plot.title("Low Rank SVD, RMSE = " + `RMSE`)
plot.imshow(A_, cmap=cm.gray)

#Keep leaning rate constant for all examples
l = 0.001
#Sparse matrix, gradient descent example
#for b in [0.,.1,.2.,.3,.4,.5,.6,.7,.8,.9,1]:
for b in [.3]:
    A_=PMF(A,approx=K,iterations=I,regularization_rate=b,learning_rate=l)
    plot.figure()
    RMSE = np.sum(np.sum(A-A_))/float(A.shape[0]*A.shape[1])
    plot.title("PMF, RMSE = " + `RMSE`)
    plot.imshow(A_, cmap=cm.gray)

#Sparse matrix, kernelized probabilistic matrix
#for b in [0.,.1,.2.,.3,.4,.5,.6,.7,.8,.9,1]:
for b in [.3]:
    A_=KPMF(A,approx=K,iterations=I,adjacency_width=2,adjacency_strength=b,learning_rate=l)
    plot.figure()
    RMSE = np.sum(np.sum(A-A_))/float(A.shape[0]*A.shape[1])
    plot.title("Kernelized PMF, $\beta$ = " + b + ", RMSE = " + `RMSE`)
    plot.imshow(A_, cmap=cm.gray)

#Sparse matrix, kernelized probabilistic matrix
#A_=KPMF(A,approx=K,iterations=I,adjacency_width=2,adjacency_strength=.2)
#plot.figure()
#plot.title("Kernelized PMF, RMSE = ")
#plot.imshow(A_, cmap=cm.gray)
#plot.show()
