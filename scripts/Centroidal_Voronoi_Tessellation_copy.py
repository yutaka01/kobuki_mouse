#! /usr/bin/env python
# coding: UTF-8
import random
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
 

def centroidal(vor, pts):
    sq = Polygon([[0, 0], [10, 0], [10, 10], [0, 10]])
    maxd = 0.0
    for i in range(len(pts) - 3):
        poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        i_cell = sq.intersection(Polygon(poly)) #sq&Polygon(poly)と等価
        p = Point(pts[i])
        pts[i] = i_cell.centroid.coords[0]
        d = p.distance(Point(pts[i]))
        if maxd < d: maxd = d
    return maxd


if __name__ == '__main__':
    pts = [[2,7],[6,9],[9,8],[7,6],[8,3],[5,4],[4,2],[1,1],[3,3],[5,5]]
    pts = pts + [[100, 100], [100, -100], [-100, 0]]

    plt.figure(figsize=(6, 6))
    d_threshold = 0.001
    for i in range(100):
        vor = Voronoi(pts)
        d = centroidal(vor, pts)


        plt.cla()
        voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False)

        plt.gca().set_aspect('equal')
        plt.gca().set_xlim([0, 10])
        plt.gca().set_ylim([0, 10])
        plt.savefig(str(i).zfill(2) + 'beta.png', bbox_inches='tight')

        if d < d_threshold:
            break
    plt.show()
