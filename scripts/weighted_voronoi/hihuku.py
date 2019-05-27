#! /usr/bin/env python
# coding: UTF-8

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, ode
from scipy.optimize.slsqp import approx_jacobian
from functools import partial


def phi(X, Y, p):
    #print(p)

    hyolist = []
    for i in range(5):
        hyo = ((X - p[0][i]) ** 2 + (Y - p[1][i]) ** 2)
        hyolist.append(hyo)
    #print(min(hyolist))
    a = 1
    #print(min(hyolist))
    global num
    num += 1
    #print(num)
    return min(hyolist)

def J(p):
    #print(p)
    #print(p[1][2])
    a =  dblquad(lambda X, Y: phi(X, Y, p), 0, 1, 0, 1)
    #print(a[0])
    b = a[0]
    return b

def dJdp(p):
    #partial(J, p)
    #vJ = np.vectorize(J)
    #print(len(p))
    #print(p)
    return approx_jacobian(p, J(p), 0.000000001)


    num = 0
    n = 5
    #p = [[random.random(), random.random()] for i in  range(n)]
    values = np.random.rand(2*5, 1)
    p = 0.2*values
    P = np.array(p).T
    a = 1
    p = p.reshape((2, 5), order='F')
    print(p)
    #vJ = np.vectorize(J)
    #print(type(vJ))
    #print(vJ(p))
    #phi =  min(np.sqrt(sum((q - (np.reshape(p, (2, n), order='F')) ** 2))))
    #J = dblquad(q: phi(x, y), 0, 1, 0, 1)
    print(dJdp(p))
    end = 0
