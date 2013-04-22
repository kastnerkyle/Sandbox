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
approx = 50
sparseness = .9
A = lena()
U,S,VT = np.linalg.svd(A, full_matrices=False)
A_ = np.zeros((len(U), len(VT)))
for i in xrange(approx):
    A_ += S[i]*np.outer(U.T[i],VT[i])
plot.imshow(A, cmap=cm.gray)
plot.figure()
plot.imshow(A_, cmap=cm.gray)
plot.show()

#Sparse matrix, regular SVD example
approx = 50
sparseness = .9
A = lena()
for i in xrange(len(A)):
    for j in xrange(len(A[i])):
        if np.random.rand() < sparseness:
            A[i][j] = 0.
U,S,VT = np.linalg.svd(A, full_matrices=False)
A_ = np.zeros((len(U), len(VT)))
for i in xrange(approx):
    A_ += S[i]*np.outer(U.T[i],VT[i])
plot.imshow(A, cmap=cm.gray)
plot.figure()
plot.imshow(A_, cmap=cm.gray)
plot.show()
