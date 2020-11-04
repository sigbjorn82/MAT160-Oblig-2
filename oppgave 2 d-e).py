import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

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
    x_norm = []
    e_norm = []

    for i in range(n):
        x = np.dot(inv(D), (b - np.dot(LU,x)))
        x_iter.append(x)
        e = abs(np.dot(A,x)-b)
        e_i.append(e)

        e_norm.append(norm(e))
        x_norm.append(norm(x))

    return pd.DataFrame({'x Approximations jacobi': x_iter}), pd.DataFrame({'errors': e_i}), pd.DataFrame({'error_norm': e_norm}), pd.DataFrame({'x_norm': x_norm})

def gaus_seidel_brutalforce(A, b, x_0, n):
    D =  np.diag(np.diag(A))
    U = np.triu(A)-D
    L = np.tril(A)
    x_iter = []
    e_i = []
    x_norm = []
    e_norm = []
    for i in range(n):
        x_new = np.dot(inv(L), (b - np.dot(U, x_0)))
        x_iter.append(x_new)
        x_0 = x_new
        e = abs(np.dot(A,x_new)-b)

        e_i.append(e)
        e_norm.append(norm(e))
        x_norm.append(norm(x_new))
    return pd.DataFrame({'x Approximations gaus_seidel_Brutal_Force': x_iter}), pd.DataFrame({'errors': e_i}), pd.DataFrame({'error_norm': e_norm}), pd.DataFrame({'x_norm': x_norm})

from matplotlib.scale import LogScale
def gaus_seidel_backsub(A, b, x_0, n):
    D =  np.diag(np.diag(A))
    U = np.triu(A)
    L = np.tril(A)-D
    x_iter = []
    e_i = []
    x_norm = []
    e_norm = []
    for i in range(n):

        x_new = backU(U= U, b= (b-np.dot(L,x_0)), N= A.shape[0])
        x_iter.append(x_new)
        x_0 = x_new
        e = abs(np.dot(A,x_new)-b)

        e_i.append(e)
        e_norm.append(norm(e))
        x_norm.append(norm(x_new))

    return pd.DataFrame({'x Approximations gaus_seidel_BackSubst': x_iter}), pd.DataFrame({'errors': e_i}), pd.DataFrame({'error_norm': e_norm}), pd.DataFrame({'x_norm': x_norm})



# Oppgave 2c)

Alpha_1 = 0.1
Alpha_2 = 0.2
Alpha_3 = 0.5
Alpha_4 = 2.0

A_alpha = lambda alpha: np.matrix(sp.diags([-1, (2+alpha), -1], [1, 0, -1], shape=(1000, 1000)).toarray()) #1000 X 1000 MATRIX

b_alpha = np.ones((1000, 1))

x0 =np.zeros((1000, 1), dtype='float')

dfGS1 = gaus_seidel_brutalforce(A_alpha(Alpha_1), b_alpha, x0, n=40)[2].cumsum()
dfGS2 = gaus_seidel_brutalforce(A_alpha(Alpha_2), b_alpha, x0, n=40)[2].cumsum()
#dfGS3 = gaus_seidel_brutalforce(A_alpha(Alpha_3), b_alpha, x0, n=200)[2].cumsum()
#dfGS4 = gaus_seidel_brutalforce(A_alpha(Alpha_4), b_alpha, x0, n=200)[2].cumsum()

plt.figure('Gaus-Seidell')

plt.subplot()
plt.plot(dfGS1)
plt.plot(dfGS2)
#plt.plot(dfGS3)
#plt.plot(dfGS4)
plt.grid(True)
plt.yscale('log')

plt.show()


#plot Jacobi
dfJ1 = jacobi(A_alpha(Alpha_1), b_alpha, x0, n=40)[2].cumsum()
dfJ2 = jacobi(A_alpha(Alpha_1), b_alpha, x0, n=40)[2].cumsum()
#dfJ3 = jacobi(A_alpha(Alpha_1), b_alpha, x0, n=200)[2].cumsum()
#dfJ4 = jacobi(A_alpha(Alpha_1), b_alpha, x0, n=200)[2].cumsum()

plt.figure('jacobi')

plt.subplot()
plt.plot(dfJ1)
plt.plot(dfJ2)
#plt.plot(dfJ3)
#plt.plot(dfJ4)

plt.grid(True)
plt.yscale('log')

plt.show()


