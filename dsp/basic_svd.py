#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plot

#Implementation of example from
#Singular Value Decomposition Tutorial
#Kirk Baker
#pg 17

def get_sort_indices(arr):
    """Returns the indices from a descending order sort"""
    return [i[0] for i in sorted(enumerate(arr),reverse=True,key=lambda x: x[1])]

def order_matrix(eigs):
    M = np.asmatrix(np.zeros(eigs[1].shape))
    for n,i in enumerate(get_sort_indices(eigs[0])):
        M[:,i] = eigs[1][:,n]
    return M

def order_eigvals(eigs):
    non_zero = eigs[0][np.where(eigs[0] != 0)[0]]
    M = np.asmatrix(np.zeros((len(non_zero), len(non_zero)+1)))
    for n,i in enumerate(sorted(non_zero, reverse=True)):
        M[n,n] = np.sqrt(i)
    return M
A = np.asmatrix([[3,1,1],[-1, 3, 1]])
transpose_first = A.T*A
transpose_last = A*A.T
#Tutorial mentions doing Graham-Schmidt Orthonormalization but it looks
#like numpy does this already - awesome!
#Output is an array of [eigenvalues, eigenvectors]
transpose_first_eigs = np.linalg.eig(A.T*A)
transpose_last_eigs = np.linalg.eig(A*A.T)
print A
U = order_matrix(transpose_last_eigs)
print U
V = order_matrix(transpose_first_eigs)
print V
S = order_eigvals(transpose_first_eigs)
print S
print U*S*V.T
print "BROKEN!!!!! Eigenvector decomposition is not unique (sign can change)..."
