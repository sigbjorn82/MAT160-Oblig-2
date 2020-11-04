import numpy as np
from numpy.linalg import inv
import pandas as pd
import scipy.sparse as sp
import matplotlib


def backU(U,b,N):
    '''Takes inn triangular matrix U, vector b and dimention of matrix n
    computes x from matrix equation Ax=b troughout nested backsubstitution'''

    x_computed = np.zeros(N)
    for i in range(N - 1, -1, -1):                  # itererer over matrisen vertikalt
        x_tmp = b[i]                                # henter ut siste kjente x

        for j in range(N - 1, i, -1):               # iterer over kollonene for neste x gitt x_temp = kollonens b
            x_tmp =x_tmp - x_computed[j] * U[i, j]  # beregner neste x

        x_computed[i] = x_tmp / U[i, i]
    return x_computed


def jacobi(A, b, x, n):
    D =  np.diag(np.diag(A))
    U = np.triu(A)-D
    L = np.tril(A)-D
    LU = L+U
    x_iter = []
    e_i = []

    for i in range(n):
        x = np.dot(inv(D), (b - np.dot(LU,x)))
        x_iter.append(x)
        e = abs(np.dot(A,x)-b)
        e_i.append(e)
    return pd.DataFrame({'x Approximations jacobi': x_iter}), pd.DataFrame({'errors': e_i})

def gaus_seidel_brutalforce(A, b, x_0, n):
    D =  np.diag(np.diag(A))
    U = np.triu(A)-D
    L = np.tril(A)
    x_iter = []
    e_i = []
    for i in range(n):
        x_new = np.dot(inv(L), (b - np.dot(U, x_0)))
        x_iter.append(x_new)
        x_0 = x_new


        e = abs(np.dot(A,x_new)-b)
        e_i.append(e)

    return pd.DataFrame({'x Approximations gaus_seidel_Brutal_Force': x_iter}), pd.DataFrame({'errors': e_i})

def gaus_seidel_backsub(A, b, x_0, n):
    D =  np.diag(np.diag(A))
    U = np.triu(A)
    L = np.tril(A)-D
    x_iter = []
    e_i = []
    for i in range(n):

        x_new = backU(U= U, b= (b-np.dot(L,x_0)), N= A.shape[0])
        x_iter.append(x_new)
        x_0 = x_new

        e = abs(np.dot(A,x_new)-b)
        e_i.append(e)

    return pd.DataFrame({'x Approximations gaus_seidel_BackSubst': x_iter}), pd.DataFrame({'errors': e_i})



# Oppgave 2c)

Alpha_1 = 0.1
Alpha_2 = 0.2
Alpha_3 = 0.5
Alpha_4 = 2.0

A_alpha = lambda alpha: np.matrix(sp.diags([-1, (2+alpha), -1], [1, 0, -1], shape=(1000, 1000)).toarray()) #1000 X 1000 MATRIX

b_alpha = np.ones((1000, 1))

x0 =np.zeros((1000, 1), dtype='float')

df=gaus_seidel_brutalforce(A_alpha(Alpha_1), b_alpha, x0, n=10)[1]

print(df)


