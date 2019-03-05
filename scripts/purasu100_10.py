#! /usr/bin/env python
import random , numpy ,rospy
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
#100×10、三台の重心ボロノイ図計算
class Centoro:
    @classmethod
    def centroidal(self,vor, pts):
        sq = Polygon([[0, 0], [100, 0], [100, 10], [0, 10]])
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
    n = 3
    pts = [[random.uniform(0,100), random.uniform(0,10)] for i in range(n)]
    for v in range(n):
        a = v
        print('X%d=' %(v+1))
        print(pts[v])#v' = 'pts[v])
    pts = pts + [[100, 100], [100, -100], [-100, 0]]
    #pts = pts.replace('"', '')
    plt.figure(figsize=(6, 6))
    d_threshold = 0.005
    num = 0
    while True:
        
        num += 1
        vor = Voronoi(pts)
        d = Centoro.centroidal(vor, pts)
 
        plt.cla()
        fig = voronoi_plot_2d(vor, ax=plt.gca(),show_vertices=False) #ax=plt.gca()

        plt.gca().set_aspect('equal')
        plt.gca().set_xlim([0, 100])
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
    plt.show()


