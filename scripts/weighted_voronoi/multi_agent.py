#! /usr/bin/env python
# coding: UTF-8

import numpy as np
import autograd.numpy as np
from autograd import grad, jacobian
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import sympy as sym
import math
import drawnow
import functools
import operator


if __name__ == '__main__':
    N = 5
    values = np.random.rand(2*5, 1)
    p = 0.2*values
    p = np.array(p)
    P = np.conjugate(p.T)
    alpha = 1
    phi = lambda q, p: min(np.sqrt(sum((q - p.reshape((2, N), order='F')**2))))
    phiv = lambda X, Y, p: map( lambda x, y: functools.partial(phi, int(p)-1), [X, Y])
    J = lambda p: dblquad( lambda X, Y: phiv[int(X), int(Y)-1, int(p)-1], a = 0, b = 1, gfun = 0, hfun = 1)
    dJdp = lambda p: jacobian( lambda p: J[int(p)-1])

    for k in range(100):
        plt.plot(P[:, 0:-2:2], P[:, 1::2], '.-')
        drawnow
        #print(dJdp(p))
        p = p - alpha * dJdp[int(p)-1]
        P[int((0 + 1.)) - 1, :] = p
