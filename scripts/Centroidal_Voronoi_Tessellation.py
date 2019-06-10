#! /usr/bin/env python
# coding: UTF-8
import random
import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
import scipy.misc
 
def rasterize(V):
    """
    Polygon rasterization (scanlines).

    Given an ordered set of vertices V describing a polygon,
    return all the (integer) points inside the polygon.
    多角形を記述する頂点Vの順序付き集合が与えられた場合、多角形内のすべての（整数）点を返します。
    See http://alienryderflex.com/polygon_fill/

    Parameters:
    -----------

    V : (n,2) shaped numpy array
        Polygon vertices
    """

    n = len(V)
    X, Y = V[:, 0], V[:, 1]
    ymin = int(np.ceil(Y.min()))
    ymax = int(np.floor(Y.max()))
    #ymin = int(np.round(Y.min()))
    #ymax = int(np.round(Y.max()))
    P = []
    for y in range(ymin, ymax+1):
        segments = []
        for i in range(n):
            index1, index2 = (i-1) % n, i
            y1, y2 = Y[index1], Y[index2]
            x1, x2 = X[index1], X[index2]
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1
            elif y1 == y2:
                continue
            if (y1 <= y < y2) or (y == ymax and y1 < y <= y2):
                segments.append((y-y1) * (x2-x1) / (y2-y1) + x1)

        segments.sort()
        for i in range(0, (2*(len(segments)//2)), 2):
            x1 = int(np.ceil(segments[i]))
            x2 = int(np.floor(segments[i+1]))
            # x1 = int(np.round(segments[i]))
            # x2 = int(np.round(segments[i+1]))
            P.extend([[x, y] for x in range(x1, x2+1)])
    if not len(P):
        return V
    return np.array(P)

def weighted_centroid(V, D):
    """
    Given an ordered set of vertices V describing a polygon,
    return the surface weighted centroid according to density D.
    多角形を記述する頂点Vの順序付き集合が与えられた場合、密度Dに従って表面重み付き重心を返します。

    This works by first rasterizing the polygon and then
    finding the center of mass over all the rasterized points.
    これは、最初に多角形をラスタライズしてから、ラスタライズされたすべての点の重心を見つけることによって機能します。
    """

    P = rasterize(V)
    Pi = P.astype(int)
    Pi[:, 0] = np.minimum(Pi[:, 0], D.shape[1]-1)
    Pi[:, 1] = np.minimum(Pi[:, 1], D.shape[0]-1)
    D = D[Pi[:, 1], Pi[:, 0]].reshape(len(Pi), 1) #https://note.nkmk.me/python-numpy-reshape-usage/
    return ((P*D)).sum(axis=0) / D.sum()

def centroidal(vor, pts, D):
    sq = Polygon([[0, 0], [100, 0], [100, 100], [0, 100]])
    maxd = 0.0
    for i in range(len(pts) - 3):
        poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        i_cell = sq.intersection(Polygon(poly)) #sq&Polygon(poly)と等価
        alpha = list(i_cell.exterior.coords)
        alpha = np.array(alpha)
        #rasta = rasterize(alpha)
        #print(len(rasta))
        #for j in range(len(rasta)):
            #plt.plot(rasta[j][0], rasta[j][1], 'o')
            #plt.gca().set_xlim([0, 100])
            #plt.gca().set_ylim([0, 100])
        #plt.show()
        p = Point(pts[i])
        #pts[i] = i_cell.centroid.coords[0]
        pts[i] = weighted_centroid(alpha, D)
        d = p.distance(Point(pts[i]))
        if maxd < d: maxd = d
    return maxd

def phi(Q, P):
    return np.min(((np.dot((Q - P).T,  (Q - P)))**2))

def J(P):
    alpha = dblquad(lambda x, y: phi([x, y], P), 0, 100, 0, 100)
    return alpha[0]

def normalize(D):
    Vmin, Vmax = D.min(), D.max()
    if Vmax - Vmin > 1e-5:
        D = (D - Vmin) / (Vmax - Vmin)
    else:
        D = np.zeros_like(D)
    return D
 
if __name__ == '__main__':
    n = 50
    pts = [[random.randint(1,99), random.randint(1,99)] for i in range(n)]
    print(pts)
    pts = pts + [[10000, 10000], [10000, -10000], [-10000, 0]]

    plt.figure(figsize=(6, 6))
    d_threshold = 0.0001
    num = 0
    hyo = []
    D = scipy.misc.imread('mudai.png', flatten=True, mode='L')
    D = 1.0 - normalize(D)
    D = D[::-1, :]
    D = D + 0.00000001
    for i in range(100):
        num += 1
        vor = Voronoi(pts)
        print(num)
        d = centroidal(vor, pts, D)
        plt.cla()
        voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False)
        
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim([0, 100])
        plt.gca().set_ylim([0, 100])
        #plt.show()

        #plt.savefig(str(i).zfill(2) + 'alpha.png', bbox_inches='tight')
        """"
        for i in range(53):
            p.append(list(pts[i]))
        p = p[:50]
        p = np.array(p)
        hyo.append(J(p))
        print(hyo)
        p = pts[:50]
        p = np.array(p)
        hyo.append(J(p))
        print(hyo)
        """
        #if num == 1:
            #(J(p))
        #if d < d_threshold:
            #print(J(p))
            #break
    #print(J(p))

    plt.show()
    x = list(range(num))
    plt.plot(x, hyo)
    plt.show()

