#! /usr/bin/env python
# coding:utf-8
import random , numpy ,rospy
from gazebo_msgs.srv import GetModelState
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
#from sympy import *
 
class Centoro:
    @classmethod
    def centroidal(self,vor, pts):
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

    def show_gazebo_models(self):
        try:
            model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            pts = []
            name = []
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
                name.append(blockName)
            print(pts)
            print(name)
            pts = pts + [[100, 100], [100, -100], [-100, 0]]
            plt.figure(figsize=(6, 6))
            d_threshold = 0.001
            num = 0
            mydict= dict()
            while True:
        
                num += 1
                vor = Voronoi(pts)
                d = Centoro.centroidal(vor, pts)
 
                plt.cla()
                fig = voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False) #ax=plt.gca()

                plt.gca().set_aspect('equal')
                plt.gca().set_xlim([0, 10])
                plt.gca().set_ylim([0, 10])
                plt.savefig(str(num) + '.png', bbox_inches='tight')

                if d < d_threshold:
                    plt.savefig(str(num) + '.png', bbox_inches='tight')
                    break
            for v in range(5):
                print(name[v])
                print(pts[v])#v' = 'pts[v])
                mydict[name[v]] = pts[v]
                print(mydict)
            print(mydict["r1"][0])
            print(num)
            print(pts)
            plt.show()
        except rospy.ServiceException as e:
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))
            
if __name__ == '__main__':
    tuto = Tutorial()
    tuto.show_gazebo_models()

    


