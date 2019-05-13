#! /usr/bin/env python
# coding: UTF-8
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.misc
import scipy.ndimage
import numpy as np
import scipy.spatial
from shapely.geometry import Polygon, Point
 

def centroidal(vor, pts):
    sq = Polygon([[0, 0], [952, 0], [952, 952], [0, 952]])
    maxd = 0.0
    for i in range(len(pts) - 3):
        poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        i_cell = sq.intersection(Polygon(poly))
        p = Point(pts[i])
        pts[i] = i_cell.centroid.coords[0]
        d = p.distance(Point(pts[i]))
        if maxd < d: maxd = d
    return maxd


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
    # ymin = int(np.round(Y.min()))
    # ymax = int(np.round(Y.max()))
    P = []
    for y in range(ymin, ymax + 1):
        segments = []
        for i in range(n):
            index1, index2 = (i - 1) % n, i
            y1, y2 = Y[index1], Y[index2]
            x1, x2 = X[index1], X[index2]
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1
            elif y1 == y2:
                continue
            if (y1 <= y < y2) or (y == ymax and y1 < y <= y2):
                segments.append((y - y1) * (x2 - x1) / (y2 - y1) + x1)

        segments.sort()
        for i in range(0, (2 * (len(segments) // 2)), 2):
            x1 = int(np.ceil(segments[i]))
            x2 = int(np.floor(segments[i + 1]))
            # x1 = int(np.round(segments[i]))
            # x2 = int(np.round(segments[i+1]))
            P.extend([[x, y] for x in range(x1, x2 + 1)])
    if not len(P):
        return V
    return np.array(P)


def rasterize_outline(V):
    """
    Polygon outline rasterization (scanlines).

    Given an ordered set of vertices V describing a polygon,
    return all the (integer) points for the polygon outline.
    多角形を記述する頂点Vの順序付き集合を考えて、多角形アウトラインのすべての（整数）点を返します。
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
    points = np.zeros((2 + (ymax - ymin) * 2, 3), dtype=int)
    index = 0
    for y in range(ymin, ymax + 1):
        segments = []
        for i in range(n):
            index1, index2 = (i - 1) % n, i
            y1, y2 = Y[index1], Y[index2]
            x1, x2 = X[index1], X[index2]
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1
            elif y1 == y2:
                continue
            if (y1 <= y < y2) or (y == ymax and y1 < y <= y2):
                segments.append((y - y1) * (x2 - x1) / (y2 - y1) + x1)
        segments.sort()
        for i in range(0, (2 * (len(segments) // 2)), 2):
            x1 = int(np.ceil(segments[i]))
            x2 = int(np.ceil(segments[i + 1]))
            points[index] = x1, x2, y
            index += 1
    return points[:index]


def weighted_centroid_outline(V, P, Q):
    """
    多角形を記述する順序付けられた頂点Vの集合が与えられたとき、密度P＆Qに従って表面重み付き重心を返します。

    P & Q are computed relatively to density:
    P＆Qは密度に対して相対的に計算されます。
    density_P = density.cumsum(axis=1)https://qiita.com/Sa_qiita/items/fc61f776cef657242e69
    density_Q = density_P.cumsum(axis=1)

    これは、最初に多角形をラスタライズしてから、ラスタライズされたすべての点の重心を見つけることによって機能します。
    """

    O = rasterize_outline(V)
    X1, X2, Y = O[:, 0], O[:, 1], O[:, 2]

    Y = np.minimum(Y, P.shape[0] - 1)
    X1 = np.minimum(X1, P.shape[1] - 1)
    X2 = np.minimum(X2, P.shape[1] - 1)

    d = (P[Y, X2] - P[Y, X1]).sum()
    x = ((X2 * P[Y, X2] - Q[Y, X2]) - (X1 * P[Y, X1] - Q[Y, X1])).sum()
    y = (Y * (P[Y, X2] - P[Y, X1])).sum()
    if d:
        return [x / d, y / d]
    return [x, y]


def uniform_centroid(V):
    """
    多角形を記述する順序付けられた頂点Vの集合が与えられた場合、一様な表面重心を返します。

    See http://paulbourke.net/geometry/polygonmesh/
    """
    A = 0
    Cx = 0
    Cy = 0
    for i in range(len(V) - 1):
        s = (V[i, 0] * V[i + 1, 1] - V[i + 1, 0] * V[i, 1])
        A += s
        Cx += (V[i, 0] + V[i + 1, 0]) * s
        Cy += (V[i, 1] + V[i + 1, 1]) * s
    Cx /= 3 * A
    Cy /= 3 * A
    return [Cx, Cy]


def weighted_centroid(V, D):
    """
    多角形を記述する頂点Vの順序付き集合が与えられた場合、密度Dに従って表面重み付き重心を返します。

    これは、最初に多角形をラスタライズしてから、ラスタライズされたすべての点の重心を見つけることによって機能します。
    """

    P = rasterize(V)
    Pi = P.astype(int)
    Pi[:, 0] = np.minimum(Pi[:, 0], D.shape[1] - 1)
    Pi[:, 1] = np.minimum(Pi[:, 1], D.shape[0] - 1)
    D = D[Pi[:, 1], Pi[:, 0]].reshape(len(Pi), 1)  # https://note.nkmk.me/python-numpy-reshape-usage/
    return ((P * D)).sum(axis=0) / D.sum()


# http://stackoverflow.com/questions/28665491/...
#    ...getting-a-bounded-polygon-coordinates-from-voronoi-cells
def in_box(points, bbox):
    return np.logical_and(
        np.logical_and(bbox[0] <= points[:, 0], points[:, 0] <= bbox[1]),
        np.logical_and(bbox[2] <= points[:, 1], points[:, 1] <= bbox[3]))


def voronoi(points, bbox):
    # See http://stackoverflow.com/questions/28665491/...
    #   ...getting-a-bounded-polygon-coordinates-from-voronoi-cells
    # See also https://gist.github.com/pv/8036995

    # Select points inside the bounding box
    i = in_box(points, bbox)

    # Mirror points
    points_center = points[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bbox[0] - (points_left[:, 0] - bbox[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bbox[1] + (bbox[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bbox[2] - (points_down[:, 1] - bbox[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bbox[3] + (bbox[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left, points_right, axis=0),
                                 np.append(points_down, points_up, axis=0),
                                 axis=0), axis=0)
    # Compute Voronoi
    vor = scipy.spatial.Voronoi(points)

    # Filter regions
    epsilon = 0.1
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not (bbox[0] - epsilon <= x <= bbox[1] + epsilon and
                        bbox[2] - epsilon <= y <= bbox[3] + epsilon):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions
    return vor


def centroids(points, density, density_P=None, density_Q=None):
    """
    Given a set of point and a density array, return the set of weighted
    centroids.
    一組の点と密度配列が与えられたら、一組の加重重心を返します。
    """
    X, Y = points[:, 0], points[:, 1]
    # You must ensure:
    #   0 < X.min() < X.max() < density.shape[0]
    #   0 < Y.min() < Y.max() < density.shape[1]

    xmin, xmax = 0, density.shape[1]
    ymin, ymax = 0, density.shape[0]
    bbox = np.array([xmin, xmax, ymin, ymax])
    vor = voronoi(points, bbox)
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
    return np.array(centroids)


if __name__ == '__main__':
    global points
    density = scipy.misc.imread('gradient.png', flatten=True, mode='L')
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imread.html
    density = 1.0 - normalize(density)
    density = density[::-1, :]
    density_P = density.cumsum(axis=1)  # 累積和
    density_Q = density_P.cumsum(axis=1)  # 累積和

    n = 20
    points = [[random.randrange(952), random.randrange(952)] for i in range(n)]
    points = points + [[10000, 10000], [10000, -10000], [-10000, 0]]
    points = np.array(points)
    plt.figure(figsize=(6, 6))
    d_threshold = 0.001
    num = 0
    print(type(points))
    for i in range(100):
        vor = Voronoi(points)
        d = centroids(points, density)
        d2 = centroidal(vor, d)
        num += 1
        plt.cla()
        voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False)

        plt.gca().set_aspect('equal')
        plt.gca().set_xlim([0, 952])
        plt.gca().set_ylim([0, 952])
        plt.plot(d[:,0], d[:,1], 'o')
        #plt.show()
        #plt.savefig(str(i).zfill(2) + '.png', bbox_inches='tight')
        #if 30 < num:
            #break
    plt.show()