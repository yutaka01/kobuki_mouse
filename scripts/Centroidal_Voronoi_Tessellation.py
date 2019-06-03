#! /usr/bin/env python
# coding: UTF-8
import random
import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
 
 
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

