#! /usr/bin/env python
# coding: UTF-8
import random
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
 
 
def centroidal(vor, pts,n):
    sq = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    maxd = 0.0
    plt.cla()
    voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False)

    plt.gca().set_aspect('equal')
    plt.gca().set_xlim([0, 1])
    plt.gca().set_ylim([0, 1])
    plt.savefig(str(n).zfill(2) + 'bunkatu.png', bbox_inches='tight')
    for i in range(len(pts) - 3):#len(pts):ptsの要素数
        poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]] #vor.vertices:領域の頂点,vor.regions:領域
#ボロノイ領域[iのボロノイ領域のインデクス]
        i_cell = sq.intersection(Polygon(poly))
        p = Point(pts[i])#
        pts[i] = i_cell.centroid.coords[0]
        d = p.distance(Point(pts[i]))
        if maxd < d: maxd = d
    return maxd
 
if __name__ == '__main__':
    n = 20
    pts = [[random.random(), random.random()] for i in range(n)]
    #pts = pts + [[100, 100], [100, -100], [-100, 0]]
 
    plt.figure(figsize=(6, 6))
    d_threshold = 0.001
 
    for i in range(100):
	
        vor = Voronoi(pts)
        d = centroidal(vor, pts,i)
        plt.cla()
        voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False)
 
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim([0, 1])
        plt.gca().set_ylim([0, 1])
        plt.savefig(str(i).zfill(2) + '.png', bbox_inches='tight')
 
        if d < d_threshold:
            break
5
