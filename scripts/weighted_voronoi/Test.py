#! /usr/bin/env python
# coding: UTF-8

import numpy as np


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

        X = store_X[:]

        # f(x-h)
        X[i] -= h
        f_x_minus_h = f(X)

        # 偏微分
        gradient[i] = (f_x_plus_h - f_x_minus_h) / (2 * h)
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

if __name__ == '__main__':
    f = lambda X: X[0] ** 2 + X[1] ** 2
    X = np.array([3.0, 4.0])
    gradient_descent(f, X, learning_rate=0.1, max_iter=100)