
# Minimizzazione mediante decomposizione in blocchi di una variabile di funz a 4 var
# Gianmarco Santoro


import numpy as np


def f(x):
    n = len(x)
    y = np.sum((x - 1) ** 2)
    y += 4 * np.prod(x)
    y += np.prod(x ** 2)
    return y


def g(x):
    n = len(x)
    gy = np.zeros(n)
    for i in range(n):
        gy[i] += 2 * (x[i] - 1)
        p1 = 1.0
        for j in range(n):
            if j != i:
                p1 *= x[j]
        gy[i] += 4 * p1 + 2 * x[i] * p1 ** 2
    return gy


def starp(i,x):
    n = len(x)
    p1 = 1.0
    for j in range(n):
        if j != i:
            p1 *= x[j]
    y = 2 - 4 * p1
    den = 2 * (1 + p1 ** 2)
    return y/den


x = np.ones(4)


print("--------------------------------------------------------------\n"
      " k |i|     xi*     |     f(x)    |    df/dxi   |    ||∇f||    \n"
      "--------------------------------------------------------------\n")

y = x

for k in range(1, 30 + 1):  # starting and final iteration

    for i in range(4):

        xi_init = starp(i, y)
        y[i] = xi_init
        gf = g(y)
        dfdxi = gf[i]
        ngf = np.linalg.norm(gf)

        print(f'{k:>2} {i + 1:>2} {xi_init:+9.6e} {f(x):+9.6e} {dfdxi:+9.6e} {ngf:+9.6e}')

print("\n--------------------------------------------------------------\n"
      "--------------  Iteration process completed  -----------------\n"
      "--------------------------------------------------------------\n")
