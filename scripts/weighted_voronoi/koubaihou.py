#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import dblquad

def phi(Q, P):
    return np.min(np.dot((Q - P).T,  (Q - P)))

def J(P):
    alpha = dblquad(lambda x, y: phi([x, y], P), 0, 1, 0, 1)
    return alpha[0]


def numerical_diff(f, x, i):
    """中央差分を元に数値微分する関数 (偏微分)

    :param function f: 偏微分する関数
    :param numpy.ndarray x: 偏微分する引数
    :param int i: 偏微分する変数のインデックス
    """
    # 丸め誤差で無視されない程度に小さな値を用意する
    h = 1e-4
    # 偏微分する変数のインデックスにだけ上記の値を入れる
    h_vec = np.zeros_like(x)
    h_vec[i] = h
    # 数値微分を使って偏微分する
    return (f(x + h_vec) - f(x - h_vec)) / (2 * h)


def numerical_gradient(f, x):
    """勾配を計算する関数

    勾配というのは、全ての変数について偏微分した結果をベクトルとしてまとめたものを言う。
    """
    # 勾配を入れるベクトルをゼロで初期化する
    grad = np.zeros_like(x)

    for i, _ in enumerate(x):
        # i 番目の変数で偏微分する
        grad[i] = numerical_diff(f, x, i)

    # 計算した勾配を返す
    return grad


def gradient_descent(f, initial_position, learning_rate=0.1, steps=50):
    """勾配法で最小値を求める関数

    :param function f: 最小値を見つけたい関数
    :param numpy.ndarray initial_position: 関数の初期位置
    :param float learning_rate: 学習率
    :param int steps: 学習回数
    """
    # 現在地を示すカーソル
    x = initial_position

    # 学習を繰り返す
    for _ in range(steps):
        # 現在地の勾配 (どちらにどれだけ進むべきか) を得る
        grad = numerical_gradient(f, x)
        # 勾配を元にして現在地を移動する
        x -= learning_rate * grad

    # 最終的な位置を返す
    return x


def main():
    # 勾配法を使って関数 f() の最小値を探す (初期位置は 1, 2)
    min_pos = gradient_descent(f, [1, 2])
    print('勾配法が見つけた最小値: {0}'.format(min_pos))


if __name__ == '__main__':
    main()