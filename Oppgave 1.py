import numpy as np
import pandas as pd
print('oppgave 1a)')


n = 4 #int(input('Size n of matrix: n= '))
A_any= np.random.randn(n, n)
U_any = A_any*np.tri(n).T

print('')
print('Random triangular matrix: ')

dfU_any = pd.DataFrame(
U_any,
index=None,
columns=['x_1', 'x_2', 'x_3', 'x_4'])

print(dfU_any)

x_true = np.random.randn(n)

b_any = np.dot(U_any, x_true)

def backU(U,b,n):
    '''Takes inn triangular matrix U, vector b and dimention of matrix n
    computes x from matrix equation Ax=b troughout nested backsubstitution'''

    x_computed = np.zeros(n)
    for i in range(n - 1, -1, -1):                  # itererer over matrisen vertikalt
        x_tmp = b[i]                                # henter ut siste kjente x

        for j in range(n - 1, i, -1):               # iterer over kollonene for neste x gitt x_temp = kollonens b
            x_tmp =x_tmp - x_computed[j] * U[i, j]  # beregner neste x

        x_computed[i] = x_tmp / U[i, i]
    return x_computed

print('') #space
print('') #--"--
print('') #--"--


x_numpy = lambda U, b: np.linalg.solve(U, b)             # numpy sin innebygde solver

#samlet visning av resultatene
df = pd.DataFrame(
[x_true,
backU(U=U_any, b=b_any, n=n),
x_numpy(U_any,b_any)],
index=['x_true', 'x_computed', 'x_(numpy.solue)*'],
columns=['x_1', 'x_2', 'x_3', 'x_4'])

print(df)
print('* For en utvidet kontroll testes matrisen med numpys innebygde solver funksjon som ikke er backsubstitution')
print('')
print('Oppgave 1b)')

U =-np.tri(10,10).T
b= np.ones(10).T
print('U - Matrix:')
print(U)
print('')
print('b-vector:')
print(b)
print('')
print('Ux=b løst ved bruk av backU funksjon:')
print('x= ',backU(U,b,10))
print('')
print('')
print('Til sammenligning får vi ved numpy solver solution: x= ', x_numpy(U,b))

