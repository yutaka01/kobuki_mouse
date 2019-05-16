#! /usr/bin/env python
# coding: UTF-8

import numpy as np
import scipy
import matcompat
import matplotlib.pylab as plt
# if available import pylab (from matlibplot)

if __name__ == '__main__':

    N = 5.
    p = np.dot(0.2, np.random.rand(int(2.*N), 1.)
    P = p.conj().T
    alpha = 1.
    phi = lambda q, p: matcompat.max(np.sqrt(np.sum(((q-np.reshape(p, 2., N))**2.))))
    phiv = lambda X, Y, p: arrayfun(lambda x, y: phi[int(np.array(np.hstack((x, y))).conj().T)-1,int(p)-1], X, Y)
    J = lambda p: quad2d(lambda X, Y: phiv[int(X)-1,int(Y)-1,int(p)-1], 0., 1., 0., 1., 'Singular', false)
    dJdp = lambda p: numjac(lambda t, p: J[int(p)-1], 0., p, J[int(p)-1], np.dot(1e-6, np.ones((2.*N), 1.)), np.array([]), 0.)
    for k in np.arange(1., 101.0):
        plt.clf
        fcontour(lambda X, Y: phiv[int(X)-1,int(Y)-1,int(p)-1], np.array(np.hstack((0., 1., 0., 1.))), 'k', 'LevelStep', 1e-2)
        plt.axis(equal)
        plt.hold(on)
        plt.title(sprintf('k=%d', k))
        plt.plot(P[:,0:0-1.:2.], P[:,1::2.], '.-')
        drawnow
        p = p-np.dot(alpha, dJdp[int(p)-1].conj().T)
        P[int((0+1.))-1,:] = p
    
