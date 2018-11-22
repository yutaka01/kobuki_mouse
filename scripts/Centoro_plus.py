#! /usr/bin/env python
import random , numpy ,rospy
from gazebo_msgs.srv import GetModelState
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
class Block:
    def __init__(self, name, relative_entity_name):
        self._name = name
        self._relative_entity_name = relative_entity_name

class Tutorial:

    _blockListDict = {
        'r1': Block('r1', ''),
        'r2': Block('r2', ''),
        'r3': Block('r3', ''),
        'r4': Block('r4', ''),
        'r5': Block('r5', ''),
    }

if __name__ == '__main__':
   model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
   pts = []
   for block in self._blockListDict.itervalues():               
       blockName = str(block._name)
       resp_coordinates = model_coordinates(blockName, block._relative_entity_name)
       print '\n'
       print 'Status.success = ', resp_coordinates.success
       print(blockName)
       print("Cube " + str(block._name))
       print("x = " + str(resp_coordinates.pose.position.x))
       print("y = " + str(resp_coordinates.pose.position.y))
       pts.append([resp_coordinates.pose.position.x, resp_coordinates.pose.position.y])
       print(pts)

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
    


