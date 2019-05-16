#! /usr/bin/env python
# coding: UTF-8

import numpy as np
import autograd.numpy as np
from autograd import grad, jacobian
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import sympy as sym
import drawnow
from functools import partial
from scipy.optimize.slsqp import approx_jacobian

"""
def jacobi(A, b, tol):
    # 線形連立方程式をヤコビ法で解く
    xOld = np.empty_like(b)
    error = 1e12

    D = np.diag(A)
    R = A - np.diagflat(D)

    while error > tol:
        x = (b-np.dot(R, xOld))/D
        error = np.linalg.norm(x-xOld)/np.linalg.norm(x)
        xOld = x
    return x
"""

def phi(q, p):
    a = 0
    return min(np.sqrt(sum((q - (np.reshape(p, (2, 5), order='F')) ** 2))))

def phiv(X, Y, p):
    bubun = phi_partial(p)
    print(type(bubun))
    print(X, Y)
    return lambda x, y: map(bubun(q = np.array[x, y].T), [X, Y])

def I(p):
    bubun2 = phiv_partial(p)
    print(type(bubun2))
    print(bubun2)
    #plt.plot(bubun[:, 0:-2:2], bubun[:, 1::2], '.-')
    print(bubun2)
    return dblquad(lambda X, Y: bubun2(X = X, Y = Y), 0, 1, 0, 1)

def dJdp(p):
    return approx_jacobian(p, I(p))

def phiv_partial(p):
    return partial(phiv, p = p)

def phi_partial(p):
    return partial(phi, p = p)

if __name__ == '__main__':
    N = 5.
    values = np.random.rand(2*5, 1)
    p = 0.2*values
    P = np.array(p).T
    alpha = 1.
    #alpha = np.array(alpha)
    """
    phi = lambda q, p: min(np.sqrt(sum((q - p.reshape((2, N), order='F')**2))))
    phiv = lambda X, Y, p:(lambda x, y: phi[np.conjugate(x, y).T, p], X, Y)
    I = lambda p: scin.dblquad(lambda X, Y: phiv[X, Y, p], 0, 1, 0, 1)
    dJdp = lambda p: Jacobian(lambda t, p: I(p), 0, p, I(p), np.exp(1)-6*np.ones(2*5, 1), [], 0)
    """
    #J = jacobian(p)
    for k in range(100):
        plt.plot(P[:, 0:-2:2], P[:, 1::2], '.-')
        drawnow
        #hbibun = dJdp(p)

        p = p - alpha * (dJdp(p))
        P[int((0 + 1.)) - 1, :] = p