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
    print "Running low rank SVD with:"
    print "approximation rank=" + `K`
    print ""
    for i in xrange(K):
        A_ += S[i]*np.outer(U.T[i],VT[i])
    return A_

def PMF(input_matrix, approx=50, iterations=30, learning_rate=.001, regularization_rate=.1):
    A = input_matrix
    Z = np.asarray(A > 0,dtype=np.int)
    A1d = np.ravel(A)
    mean = np.mean(A1d)
    #Remove DC term (mean), advice from
    #http://www.intelligentmining.com/2011/08/08/intro-to-matrix-factorization/
    A = A-mean
    K = approx
    R = itr = iterations
    l = learning_rate
    b = regularization_rate
    N = A.shape[0]
    M = A.shape[1]
    U = np.random.randn(N,K)
    V = np.random.randn(K,M)
    print "Running PMF with:"
    print "learning rate=" + `l`
    print "regularization rate=" + `b`
    print "approximation rank=" + `K`
    print "iterations=" + `R`
    print ""
    #PMF using gradient descent as per paper
    #Probabilistic Matrix Factorization
    #R. Salakhutdinov, A. Minh
    for r in range(R):
        for i in range(N):
            for j in range(M):
                if Z[i,j] > 0:
                    e = A[i,j] - np.dot(U[i,:],V[:,j])
                    U[i,:] = U[i,:] + l*(e*V[:,j] - b*U[i,:])
                    V[:,j] = V[:,j] + l*(e*U[i,:] - b*V[:,j])
    A_ = np.dot(U,V)
    return A_

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
    #KPMF using gradient descent as per paper
    #Kernelized Probabilistic Matrix Factorization: Exploiting Graphs and Side Information
    #T. Zhou, H. Shan, A. Banerjee, G. Sapiro
    #Using diffusion kernel
    #U are the rows, we use an adjacency matrix CU to reprent connectivity
    #This matrix connects rows +-adjacency_width
    #V are the columns, connected columns are CV
    #Operate on graph laplacian L, which is the degree matrix D - C
    #Applying the diffusion kernel to L, this forms a spatial smoothness graph
    bw = adjacency_width
    #Use scipy.sparse.diags to generate band matrix with bandwidth = 2*adjacency_width+1
    #Example of adjacency_width = 1, N = 4
    #[1 1 0 0]
    #[1 1 1 0]
    #[0 1 1 1]
    #[0 0 1 1]
    CU = sp.diags([1]*(2*bw+1),range(-bw,bw+1),shape=(N,N)).todense()
    DU = np.diagflat(np.sum(CU,1))
    CV = sp.diags([1]*(2*bw+1),range(-bw,bw+1),shape=(M,M)).todense()
    DV = np.diagflat(np.sum(CV,1))
    LU = DU - CU
    LV = DV - CV
    beta = adjacency_strength
    KU = sl.expm(beta*LU)
    KV = sl.expm(beta*LV)
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

def get_RMSE(A,A_):
    A1d = np.ravel(A)
    A_1d = np.ravel(A_)
    e = np.mean((A1d-A_1d)**2)
    return np.sqrt(e)

#Rework of lena example
#from https://gist.github.com/thearn/5424219
#Sparse lena
A = np.asarray(lena(),dtype=np.double)
plot.figure()
plot.title("Original Lena")
plot.imshow(A, cmap=cm.gray)
plot.savefig("pristine_lena.png")

#Full matrix SVD, low rank approximation
approx = K = 10
iterations = I = 7
origA = lena()
A_= lowrank_SVD(A,approx=K)
plot.figure()
RMSE = get_RMSE(origA,A_)
plot.title("Low Rank SVD (full matrix)\nRMSE=" + `RMSE`)
plot.imshow(A_, cmap=cm.gray)
plot.savefig("SVD_full_approx_"+`K`+".png")

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
plot.savefig("sparse_lena_s_"+`sparseness`+".png")

#Sparse matrix, regular SVD example, low rank approximation
A_=lowrank_SVD(A,approx=K)
plot.figure()
RMSE = get_RMSE(origA,A_)
plot.title("Low Rank SVD\nRMSE="+`RMSE`)
plot.imshow(A_, cmap=cm.gray)
plot.savefig("SVD_sparse_approx_"+`K`+".png")

#Keeping learning rate constant for all PMF examples
l = 0.001
#Sparse matrix, gradient descent example
#Save RMSE as tuple of two values, (b,RMSE)
PMF_RMSE=[]
for b in [0.,.25,.5,.75,1.]:
    A_=PMF(A,approx=K,iterations=I,regularization_rate=b,learning_rate=l)
    plot.figure()
    RMSE = get_RMSE(origA,A_)
    PMF_RMSE.append((b,RMSE))
    #Lambda is the regularization rate (lambda_v = lambda_u from the paper
    plot.title("PMF, $\lambda$=" + `b` + "\nRMSE=" + `RMSE`)
    plot.imshow(A_, cmap=cm.gray)
    plot.savefig("PMF_b_"+`int(10*b)`+".png")

#Sparse matrix, kernelized probabilistic matrix
#Save RMSE as tuple of 3 values (b,w,RMSE)
KPMF_RMSE=[]
for w in [0,4,10,20,40]:
    blist = []
    for b in [0.,.25,.5,.75,1.]:
        A_=KPMF(A,approx=K,iterations=I,adjacency_width=w,adjacency_strength=b,learning_rate=l)
        plot.figure()
        RMSE = get_RMSE(origA,A_)
        blist.append(RMSE)
        plot.title(r"Kernelized PMF, $\beta$=" + `b` + ", width=" + `w`+ "\nRMSE=" + `RMSE`)
        plot.imshow(A_, cmap=cm.gray)
        plot.savefig("KPMF_w"+`w`+"_b_"+`int(10*b)`+".png")
    KPMF_RMSE.append((w,tuple(blist)))

colors=['r','b','g','c','m','k']
plot.figure()
bs,RMSEs = zip(*PMF_RMSE)
ws,bRMSEs = zip(*KPMF_RMSE)
labels=["PMF"]+["KPMF_"+`w` for w in ws]
plot.title("RMSE vs. regularization/diffusion rate")
for err,col,lbl in zip((RMSEs,)+bRMSEs,colors,labels):
    plot.plot(bs, err, col, label=str(lbl))
plot.xlabel('Regularization rate')
plot.ylabel('RMSE')
plot.legend()
plot.savefig("reg_rate_and_diffusion_RMSE.png")

plot.figure()
plot.title("Average RMSE vs. kernel bandwidth")
for err, col, lbl in zip(bRMSEs,
                         colors,
                         bs):
    plot.plot(ws, err, col, label=str(lbl))
plot.xlabel('Kernel bandwidth')
plot.ylabel('Average RMSE over calculated rates')
plot.legend()
plot.savefig("kernel_width_mean_RMSE.png")
