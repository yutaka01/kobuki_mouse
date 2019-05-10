#! /usr/bin/env python3
# coding: UTF-8
import tqdm
import voronoi
import os.path
import scipy.misc
import scipy.ndimage
import numpy as np

def normalize(D):
    Vmin, Vmax = D.min(), D.max()
    if Vmax - Vmin > 1e-5:
        D = (D-Vmin)/(Vmax-Vmin)
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
        X = np.random.uniform(0, density.shape[1], 10*n)
        Y = np.random.uniform(0, density.shape[0], 10*n)
        P = np.random.uniform(0, 1, 10*n)
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
    points = np.zeros((2+(ymax-ymin)*2, 3), dtype=int)
    index = 0
    for y in range(ymin, ymax+1):
        segments = []
        for i in range(n):
            index1, index2 = (i-1) % n , i
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
            x2 = int(np.ceil(segments[i+1]))
            points[index] = x1, x2, y
            index += 1
    return points[:index]


def weighted_centroid_outline(V, P, Q):
    """
    Given an ordered set of vertices V describing a polygon,
    return the surface weighted centroid according to density P & Q.
    多角形を記述する順序付けられた頂点Vの集合が与えられたとき、密度P＆Qに従って表面重み付き重心を返します。

    P & Q are computed relatively to density:
    P＆Qは密度に対して相対的に計算されます。
    density_P = density.cumsum(axis=1)https://qiita.com/Sa_qiita/items/fc61f776cef657242e69
    density_Q = density_P.cumsum(axis=1)

    This works by first rasterizing the polygon and then
    finding the center of mass over all the rasterized points.
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
        # centroid = weighted_centroid(vertices, density)

        # Optimized version from only the outline
        centroid = weighted_centroid_outline(vertices, density_P, density_Q)

        centroids.append(centroid)
    return regions, np.array(centroids)

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    description = "Weighted Vororonoi Stippler"
    parser = argparse.ArgumentParser(description=description)

    args = parser.parse_args()

    density = scipy.misc.imread('gradient.png', flatten=True, mode='L')  # 密度分布の読み込み
    # tps://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imread.html

    # We want (approximately) 500 pixels per voronoi region
    #zoom = (args.n_point * 500) / (density.shape[0] * density.shape[1])
    #zoom = int(round(np.sqrt(zoom)))
    #density = scipy.ndimage.zoom(density, zoom, order=0)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html

    # Apply threshold onto image
    # Any color > threshold will be white
    #density = np.minimum(density, args.threshold)

    density = 1.0 - normalize(density)
    density = density[::-1, :]
    density_P = density.cumsum(axis=1)  # 累積和
    density_Q = density_P.cumsum(axis=1)  # 累積和

    #dirname = os.path.dirname(filename)
    #basename = (os.path.basename(filename).split('.'))[0]

    xmin, xmax = 0, density.shape[1]
    ymin, ymax = 0, density.shape[0]
    bbox = np.array([xmin, xmax, ymin, ymax])
    ratio = (xmax - xmin) / (ymax - ymin)

    # Interactive display
    if args.interactive:

        # Setup figure
        fig = plt.figure(figsize=(args.figsize, args.figsize / ratio),
                         facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim([xmin, xmax])
        ax.set_xticks([])
        ax.set_ylim([ymin, ymax])
        ax.set_yticks([])
        scatter = ax.scatter(points[:, 0], points[:, 1], s=1,
                             facecolor="k", edgecolor="None")


        def update(frame):
            global points
            # Recompute weighted centroids
            regions, points = voronoi.centroids(points, density, density_P, density_Q)

            # Update figure
            Pi = points.astype(int)
            X = np.maximum(np.minimum(Pi[:, 0], density.shape[1] - 1), 0)
            Y = np.maximum(np.minimum(Pi[:, 1], density.shape[0] - 1), 0)
            sizes = (args.pointsize[0] +
                     (args.pointsize[1] - args.pointsize[0]) * density[Y, X])
            scatter.set_offsets(points)
            scatter.set_sizes(sizes)
            bar.update()


        bar = tqdm.tqdm(total=args.n_iter)
        animation = FuncAnimation(fig, update,
                                  repeat=False, frames=args.n_iter - 1)
        plt.show()


    if (args.save or args.display) and not args.interactive:
        fig = plt.figure(figsize=(args.figsize, args.figsize / ratio),
                         facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim([xmin, xmax])
        ax.set_xticks([])
        ax.set_ylim([ymin, ymax])
        ax.set_yticks([])
        scatter = ax.scatter(points[:, 0], points[:, 1], s=1,
                             facecolor="k", edgecolor="None")
        Pi = points.astype(int)
        X = np.maximum(np.minimum(Pi[:, 0], density.shape[1] - 1), 0)
        Y = np.maximum(np.minimum(Pi[:, 1], density.shape[0] - 1), 0)
        sizes = (args.pointsize[0] +
                 (args.pointsize[1] - args.pointsize[0]) * density[Y, X])
        scatter.set_offsets(points)
        scatter.set_sizes(sizes)

        if args.display:
            plt.show()

    # Plot voronoi regions if you want
    # for region in vor.filtered_regions:
    #     vertices = vor.vertices[region, :]
    #     ax.plot(vertices[:, 0], vertices[:, 1], linewidth=.5, color='.5' )
