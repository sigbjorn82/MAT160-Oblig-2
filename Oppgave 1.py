import numpy as np
import array_to_latex as a2l
n = int(input('Size n of matrix: n= '))
A = np.random.randn(n, n)

a2l.to_clp(A, frmt = '{:6.2f}', arraytype = 'array')
to_tex = lambda A : a2l.to_ltx(A, frmt = '{:6.2f}', arraytype = 'array', mathform=True)
to_tex(A)
print(to_tex)
