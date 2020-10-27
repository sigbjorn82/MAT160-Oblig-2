import numpy as np
import pandas as pd



n = 4 #int(input('Size n of matrix: n= '))

A_any= np.random.randn(n, n) # Generate a random (n x n) matrix
U_any = A_any*np.tri(n).T    # Triangulating matrix


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



x_numpy = lambda U, b: np.linalg.solve(U, b)             # numpy sin innebygde solver



print('') #space
print('') #--""--
print('') #--""--




print('oppgave 1a)')
print('')
print('Random triangular matrix: ')

dfU_any = pd.DataFrame(
U_any,
index=None,
columns=['x_1', 'x_2', 'x_3', 'x_4'])

print(dfU_any)



print('') #space
print('') #--""--
print('') #--""--



#samlet visning av resultatene
df_a = pd.DataFrame(
[x_true,
backU(U=U_any, b=b_any, n=n),
x_numpy(U_any,b_any)],
index=['x_true', 'x_computed', 'x_(numpy.solue)*'],
columns=['x_1', 'x_2', 'x_3', 'x_4'])

print(df_a)
print('')
print('* For en utvidet kontroll testes matrisen med numpys innebygde solver funksjon som ikke er backsubstitution')



print('') #space
print('') #--""--
print('') #--""--


print('Oppgave 1b)')
U =-np.tri(10,10).T
b= np.ones(10).T


dfU = pd.DataFrame(
U,
index=None,
columns=['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10'])

print(dfU)



print('') #space
print('') #--""--
print('') #--""--


#samlet visning av resultatene i oppgave b
df_b = pd.DataFrame(
[backU(U,b,10), x_numpy(U,b)],
index=['x_(back substitution) ', 'x_(numpy.solue)*'],
columns=['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10'])

print(df_b)
print('')
print('* For en utvidet kontroll testes matrisen med numpys innebygde solver funksjon som ikke er backsubstitution')


print(backU(U,b,10))