#! /usr/bin/env python
import random
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
 
class Centoro:

    def centroidal(vor, pts):
        sq = Polygon([[0, 0], [10, 0], [10, 10], [0, 10]])
        maxd = 0.0
        for i in range(len(pts) - 3):
            poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
            i_cell = sq.intersection(Polygon(poly))
            p = Point(pts[i])
            pts[i] = i_cell.centroid.coords[0]
            d = p.distance(Point(pts[i]))
        if maxd < d: maxd = d
        return maxd
 
if __name__ == '__main__':
    n = 20
    pts = [[random.uniform(0,10), random.uniform(0,10)] for i in range(n)]
    for v in range(n):
        a = v
        print('X%d=' %(v+1))
        print(pts[v])#v' = 'pts[v])
    pts = pts + [[100, 100], [100, -100], [-100, 0]]
	
    plt.figure(figsize=(6, 6))
    d_threshold = 0.005
    num = 0
    while True:
        
        num += 1
        vor = Voronoi(pts)
        d = centroidal(vor, pts)
 
        plt.cla()
        fig = voronoi_plot_2d(vor, az=plt.gca(), show_vertices=False) #ax=plt.gca()

        plt.gca().set_aspect('equal')
        plt.gca().set_xlim([0, 10])
        plt.gca().set_ylim([0, 10])
        if num == 1:
            plt.savefig(str(num) + '.png', bbox_inches='tight')
        if d < d_threshold:
            plt.savefig(str(num) + '.png', bbox_inches='tight')
            break
    for v in range(n):
        print('X%d=' %(v+1))
        print(pts[v])#v' = 'pts[v])
    print(num)
    


