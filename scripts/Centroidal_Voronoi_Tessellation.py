#! /usr/bin/env python
# coding: UTF-8
import random
import numpy as np
import scipy.misc
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
 

def normalize(D):
    Vmin, Vmax = D.min(), D.max()
    if Vmax - Vmin > 1e-5:
        D = (D - Vmin) / (Vmax - Vmin)
    else:
        D = np.zeros_like(D)
    return D


def initialization(n, D):
    """
    Return n points distributed over [xmin, xmax] x [ymin, ymax]
    according to (normalized) density distribution.
    with xmin, xmax = 0, density.shape[1]
         ymin, ymax = 0, density.shape[0]
    The algorithm here is a simple rejection sampling.
    """

    samples = []
    while len(samples) < n:
        # X = np.random.randint(0, density.shape[1], 10*n)
        # Y = np.random.randint(0, density.shape[0], 10*n)
        X = np.random.uniform(0, density.shape[1], 10 * n)
        Y = np.random.uniform(0, density.shape[0], 10 * n)
        P = np.random.uniform(0, 1, 10 * n)
        index = 0
        while index < len(X) and len(samples) < n:
            x, y = X[index], Y[index]
            x_, y_ = int(np.floor(x)), int(np.floor(y))
            if P[index] < D[y_, x_]:
                samples.append([x, y])
            index += 1
    return np.array(samples)

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


def centroids(points, density, density_P=None, density_Q=None):
    """
    Given a set of point and a density array, return the set of weighted
    centroids.
    一組の点と密度配列が与えられたら、一組の加重重心を返します。
    """
    print(type(points))
    print(points[:, 0])
    X, Y = points[:, 0], points[:, 1]
    # You must ensure:
    #   0 < X.min() < X.max() < density.shape[0]
    #   0 < Y.min() < Y.max() < density.shape[1]

    xmin, xmax = 0, density.shape[1]
    ymin, ymax = 0, density.shape[0]
    bbox = np.array([xmin, xmax, ymin, ymax])
    vor = Voronoi(points)
    regions = vor.filtered_regions
    centroids = []
    for region in regions:
        vertices = vor.vertices[region + [region[0]], :]
        # vertices = vor.filtered_points[region + [region[0]], :]

        # Full version from all the points
        centroid = weighted_centroid(vertices, density)

        # Optimized version from only the outline
        # centroid = weighted_centroid_outline(vertices, density_P, density_Q)

        centroids.append(centroid)
    return regions, np.array(centroids)


def centroidal(vor, pts):
    sq = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    maxd = 0.0
    for i in range(len(pts)-3):
        poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        i_cell = sq.intersection(Polygon(poly)) #sq&Polygon(poly)と等価
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
    density = scipy.misc.imread('gradient.png', flatten=True, mode='L')
    zoom = (density.n_point * 500) / (density.shape[0] * density.shape[1])
    zoom = int(round(np.sqrt(zoom)))
    density = scipy.ndimage.zoom(density, zoom, order=0)
    density = np.minimum(density, density.threshold)

    density = 1.0 - normalize(density)
    density = density[::-1, :]
    for i in range(100):
        num += 1
        vor = Voronoi(pts)
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

