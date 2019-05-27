#! /usr/bin/env python
# coding: UTF-8

import numpy as np
import autograd.numpy as na
from autograd import grad, jacobian
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, odeint
import sympy as sym
#import drawnow
#from functools import partial
from scipy.optimize.slsqp import approx_jacobian


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


def phi(Q, P):
    return np.min(np.dot((Q - P).T,  (Q - P)))

def J(P):
    alpha = dblquad(lambda x, y: phi([x, y], P), 0, 1, 0, 1)
    return alpha[0]

def dJdp(P):
    return approx_jacobian(P, J(P), np.sqrt(np.finfo(float).eps))


if __name__ == '__main__':
    p = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.3, 0.4], [0.5, 0.5]]
    p = np.array(p)
    #P = np.conjugate(p.T)
    alpha = 1
    #h = 0.000000000001
    #for i in range(5):
        #p[i] = sym.limit((J(p + h) - J(p))/h, h, 0)
    h = 1e-4
    grad = []
    for i in range(5):
        tmp = p
        p = tmp + h
        x1 = J(p)
        p = tmp - h
        x2 = J(p)
        grad = (x2 - x1) / (2 * h)
        print(grad)
        p = tmp
    print(grad)
    print(p)
    print(p[0])
    print(p[1])
    a = 1

"""
    phi = lambda q, p: min(np.sqrt(sum((q - p.reshape((2, N), order='F') ** 2))))
    phiv = lambda X, Y, p: map(lambda x, y: functools.partial(phi, int(p) - 1), [X, Y])
    J = lambda p: dblquad(lambda X, Y: phiv[int(X), int(Y) - 1, int(p) - 1], a=0, b=1, gfun=0, hfun=1)
    dJdp = lambda p: jacobian(lambda p: J[int(p) - 1])
    for k in range(100):
        #plt.plot(P[:, 0:-2:2], P[:, 1::2], '.-')
        #drawnow
        # print(dJdp(p))
        p = p - alpha * dJdp[]
    P[int((0 + 1.)) - 1, :] = p
    """