import numpy as np
from numpy.linalg import inv
import pandas as pd
import scipy.sparse as sp

# Oppgave 2c)

alpha = 1

A_alpha = np.matrix(sp.diags([-1, (2+alpha), -1], [1, 0, -1], shape=(1000, 1000)).toarray())    #1000 X 1000 MATRIX

b_alpha = np.array(np.sum(A_alpha, axis=1))

A_alpha.sum(axis=1, dtype='float', out=np.asmatrix(b_alpha))

x0 =np.zeros((1000, 1), dtype='float')

print(A_alpha)

print('accordingto theorem 2.9 Matrix A_alpha is a strictly diagonal dominant matrix.')
print('This is that sum of row without diagonal is less than the diagonal element sum. ')
print('This leads to convergence as described in theorem 2.10. the matrix will converge')
print('to the unique solution for every vector of b in the jacobi method.')
