#! /usr/bin/env python
# coding: UTF-8

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.optimize.slsqp import approx_jacobian


def phi(X, Y, p):
    p = np.reshape(p, (2, n), order='F')
    hyolist = []
    for i in range(5):
        hyo = ((X - p[0][i]) ** 2 + (Y - p[1][i]) ** 2)
        hyolist.append(hyo)
    #print(min(hyolist))
    a = 1
    #print(min(hyolist))
    return min(hyolist)
def J(p):

    #print(p[1][2])
    a =  dblquad(lambda X, Y: phi(X, Y, p), 0, 1, 0, 1)
    print(type(a))
    return a

def dJdp(p):
    #vf = np.vectorize(J(p))
    return approx_jacobian(lambda t: p, J(p), np.sqrt(np.finfo(float).eps))

if __name__ == '__main__':
    n = 5
    #p = [[random.random(), random.random()] for i in  range(n)]
    values = np.random.rand(2*5, 1)
    p = 0.2*values
    P = np.array(p).T
    a = 1
    vJ = np.vectorize(J)
    #phi =  min(np.sqrt(sum((q - (np.reshape(p, (2, n), order='F')) ** 2))))
    #J = dblquad(q: phi(x, y), 0, 1, 0, 1)
    print(dJdp(p))
    end = 0
