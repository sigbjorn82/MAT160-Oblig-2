import random
import numpy as np
from numpy.linalg import inv
import pandas as pd

random.seed(30)

m = 3 #int(input('Size n of matrix: n= '))
A_any= np.random.randn(m, m) # Generate a random (n x n) matrix

x_true = np.random.randn(m)

b_any = np.dot(A_any, x_true)


x_0 = np.arange(m)


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
    return pd.DataFrame({'x Approximations': x_iter}), pd.DataFrame({'errors': e_i})


B = np.array([[50,2,1],[2,60,3],[1,2,70]])
b_2= np.array([1,2,3])
x0 = np.array([0,0,0])

df = jacobi(A=B, x=x0, b=b_2, n=20)

print(df)