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
    return pd.DataFrame(x_iter), pd.DataFrame(e_i)

print(jacobi(A_any, b_any, x_0, n=10)[0])
print('')
print(jacobi(A_any, b_any, x_0, n=10)[1])

