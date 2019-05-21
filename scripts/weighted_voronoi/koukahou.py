#! /usr/bin/env python
# coding: UTF-8

import random
import copy
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
    store_X = copy.deepcopy(X)
    for j in range(5):
        for i in range(2):
            # f(x+h)
            X[j, i] += h
            f_x_plus_h = f(X[j])

            X = store_X


            X[j, i] -= h

            f_x_minus_h = f(X[j])
            #f_x_minus_h = f_x_minus_h[0]


            # 偏微分
            gradient[j][i] = (f_x_plus_h[0] - f_x_minus_h[0]) / (2 * h)
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
    return X

def phi(X, Y, p):
    #print(p)
    #phi = lambda q, p: min(np.sqrt(sum((q - p.reshape((2, N), order='F')**2))))
    return ((X - p[0]) ** 2 + (Y - p[1]) ** 2)


def J(p):
    return dblquad(lambda X, Y: phi(X, Y, p), 0, 1, 0, 1)
if __name__ == '__main__':
    #values = np.random.rand(2*5, 1)
    P = [[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 4.]]
    p = np.array(P)

    f = lambda p: dblquad(lambda x, y: phi(x, y, p), 0, 1, 0, 1)
      # np.array([3.0, 4.0])
    X = p
    gradient_descent(f, X, learning_rate=0.1, max_iter=100)
    plt.plot(p)
    a = 0