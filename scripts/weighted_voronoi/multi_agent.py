#! /usr/bin/env python
# coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scin

if __name__ == '__main__':
    N = 5.
    values = np.random.rand(2*N, 1)
    p = 0.2*values
    P = np.conjugate(p.T)
    alpha = 1.
    phi = lambda q, p: min(np.sqrt(sum((q - p.reshape((2, N), order='F')**2))))
    phiv = lambda X, Y, p:(lambda x, y: phi(np.conjugate([x, y].T), p), X, Y)
    J = lambda p: scin.dblquad(lambda X, Y: phiv(X, Y, p), 0, 1, 0, 1)
    dJdp = lambda p: