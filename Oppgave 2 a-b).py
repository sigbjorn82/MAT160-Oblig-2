import random
import numpy as np
from numpy.linalg import inv
import pandas as pd

random.seed(30)

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


B = np.array([[50,2,1],[2,60,3],[1,2,70]])
b_2= np.array([1,2,3])
x0 = np.array([0,0,0])

df = jacobi(A=B, x=x0, b=b_2, n=20)



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

B = np.array([[3,1,-1],[2,4,1],[-1,2,5]])
b_2 = np.array([4,1,1])
x_guess = np.array([0,0,0])


print(gaus_seidel_backsub(A=B, b=b_2, x_0=x_guess, n=55))
print('')
print(gaus_seidel_brutalforce(A=B, b=b_2, x_0=x_guess, n=55))
print('')
