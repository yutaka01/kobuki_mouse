#! /usr/bin/env python
# coding: UTF-8

import random
import numpy as np
from scipy.integrate import dblquad, ode
import matplotlib.pyplot as plt

def calc_gradient(f, X):
    """
    calc_gradient
    偏微分を行う関数
    関数fを変数xの各要素で偏微分した結果をベクトルにした勾配を返す

    @params
    f: 対象となる関数
    X: 関数fの引数のベクトル(numpy.array)

    @return
    gradient: 勾配(numpy.array)
    """

    h = 1e-4
    gradient = np.zeros_like(X)

    # 各変数についての偏微分を計算する
    for i in range(X.size):
        store_X = X[:]

        # f(x+h)
        X[i] += h
        f_x_plus_h = f(X)
        #f_x_plus_h = f_x_plus_h[0]
        X = store_X[:]

        # f(x-h)
        X[i] -= h
        f_x_minus_h = f(X)
        #f_x_minus_h = f_x_minus_h[0]


        # 偏微分
        gradient[i] = (f_x_plus_h[0] - f_x_minus_h[0]) / (2 * h)
        print(gradient)
    return gradient


def gradient_descent(f, X, learning_rate, max_iter):
    """
    gradient_descent
    最急降下法を行う関数

    @params
    f: 対象となる関数
    X: 関数fの引数のベクトル(numpy.array)
    learning_rate: 学習率
    max_iter: 繰り返し回数

    @return
    X: 関数の出力を最小にする(であろう)引数(numpy.array)
    """

    for i in range(max_iter):
        X -= (learning_rate * calc_gradient(f, X))
        print("[{:3d}] X = {}, f(X) = {:.7f}".format(i, X, f(X)))
        #if f(X) < 0:
            #return X[i-1]
        #break
    return X

def phi(X, Y, p):
    #print(p)
    #phi = lambda q, p: min(np.sqrt(sum((q - p.reshape((2, N), order='F')**2))))
    hyolist = []
    for i in range(5):
        hyo = ((X - p[i][0]) ** 2 + (Y - p[i][1]) ** 2)
        hyolist.append(hyo)
    #print(min(hyolist))
    a = 1
    return min(hyolist)

"""
def J(p):
    print(p)
    #print(p[1][2])
    a = dblquad(lambda X, Y: phi(X, Y, p), 0, 1, 0, 1)
    a = a[0]
    print(a)
    return a
"""

if __name__ == '__main__':
    #values = np.random.rand(2*5, 1)
    P = [[random.uniform(0, 10), random.uniform(0, 10)] for i in range(5)]
    p = np.array(P)
    X = p  # np.array([3.0, 4.0])
    f = lambda p: dblquad(lambda X, Y: phi(X, Y, p), 0, 10, 0, 10)

    gradient_descent(f, p, learning_rate=0.1, max_iter=100)
    plt.plot(p)
    a = 0