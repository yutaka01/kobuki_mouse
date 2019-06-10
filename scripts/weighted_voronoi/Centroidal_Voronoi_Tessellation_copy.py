#! /usr/bin/env python
# coding: UTF-8
import random
import numpy as np
import scipy.misc
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point


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

def centroidal(vor, pts):
    sq = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    maxd = 0.0
    for i in range(len(pts)-3):
        poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        i_cell = sq.intersection(Polygon(poly)) #sq&Polygon(poly)と等価
        print(list(i_cell.exterior.coords))
        rao = np.array(i_cell)
        rasta = rasterize(i_cell)
        p = Point(pts[i])
        pts[i] = i_cell.centroid.coords[0]
        print(pts)
        d = p.distance(Point(pts[i]))
        if maxd < d: maxd = d
    return maxd

def phi(Q, P):
    wei = np.array([0.8, 0.6])
    return np.min(((np.dot((Q - P).T,  (Q - P)))**2)*(np.exp((Q - wei)/20)))

def J(P):
    alpha = dblquad(lambda x, y: phi([x, y], P), 0, 1, 0, 1)
    return alpha[0]

 
if __name__ == '__main__':
    n = 5
    pts = [[random.random(), random.random()] for i in range(n)]
    print(pts)
    pts = pts + [[100, 100], [100, -100], [-100, 0]]

    plt.figure(figsize=(6, 6))
    d_threshold = 0.0001
    num = 0
    hyo = []
    for i in range(100):
        num += 1
        vor = Voronoi(pts)
        #rasta = rasterize(vor.vertice[0])
        d = centroidal(vor, pts)

        #plt.cla()
        #voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False)
        
        #plt.gca().set_aspect('equal')
        #plt.gca().set_xlim([0, 1])
        #plt.gca().set_ylim([0, 1])
        #plt.savefig(str(i).zfill(2) + 'alpha.png', bbox_inches='tight')
        """
        for i in range(13):
            p.append(list(pts[i]))
        p = p[:10]
        p = np.array(p)
        hyo.append(J(p))
        print(hyo)
        """
        p = pts[:50]
        p = np.array(p)
        hyo.append(J(p))
        print(hyo)
        if num == 1:
            print(J(p))
        if d < d_threshold:
            print(J(p))
            break
    print(J(p))
    x = list(range(num))
    plt.plot(x, hyo)
    plt.show()

