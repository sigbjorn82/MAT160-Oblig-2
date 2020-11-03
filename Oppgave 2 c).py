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

