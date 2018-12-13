#! /usr/bin/env python
# coding:utf-8
import random ,rospy,time ,threading,tf,concurrent.futures
from geometry_msgs.msg  import Twist,Quaternion
from gazebo_msgs.srv import GetModelState
from turtlesim.msg import Pose
import matplotlib.pyplot as plt
from math import pow,atan2,sqrt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
import numpy as np
from numpy import linalg,cos
from geometry_msgs.msg import Point as Po
 
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
        #try:
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
            while True:
        
                num += 1
                vor = Voronoi(pts)
                d = Centoro.centroidal(vor, pts)
 
                plt.cla()
                fig = voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False) #ax=plt.gca()

                plt.gca().set_aspect('equal')
                plt.gca().set_xlim([0, 10])
                plt.gca().set_ylim([0, 10])
                #plt.savefig(str(num) + '.png', bbox_inches='tight')

                if d < d_threshold:
                    #plt.savefig(str(num) + '.png', bbox_inches='tight')
                    break
            global mydict
            for v in range(5):

                print(name[v])
                print(pts[v])#v' = 'pts[v])
                mydict[name[v]] = pts[v]
                print(mydict)
            print(mydict["r1"][0])
            print(num)
            print(pts)
            plt.show()
        #except rospy.ServiceException as e:
            #rospy.loginfo("Get Model State service call failed:  {0}".format(e))

class turtlebot():
        
    
    #Callback function implementing the pose value received
    def callback(self, data):
        self.pose = data
        self.pose.x = round(self.pose.x, 4)
        self.pose.y = round(self.pose.y, 4)

    def get_distance(self, goal_x, goal_y):
        distance = sqrt(pow((goal_x - self.pose.x), 2) + pow((goal_y - self.pose.y), 2)) #ノルム
        return distance

    def r1goal(self):
        print(mydict["r1"][0])
        goal_pose = Po()
        goal_pose.x = mydict["r1"][0]
        goal_pose.y = mydict["r1"][1]
        distance_tolerance = 0.1
        vel_msg = Twist()
        while True:
            _blockListDict = {
            'r1': Block('r1', ''),
            'r2': Block('r2', ''),
            'r3': Block('r3', ''),
            'r4': Block('r4', ''),
            'r5': Block('r5', ''),
            }
            #Creating our node,publisher and subscriber
            rospy.init_node('turtlebot_controller', anonymous=True)
            self.velocity_publisher = rospy.Publisher('r1/mobile_base/commands/velocity', Twist, queue_size=10)
            self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            pts = []
            name = []
            block = _blockListDict["r1"]
            blockName = str(block._name)
            self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
            #pts = (resp_coordinates.pose.position.x, resp_coordinates.pose.position.y)
            #print(resp_coordinates)
            rot_q = self.resp_coordinates.pose.orientation
            global theta
            (roll,pitch,theta) = tf.transformations.euler_from_quaternion([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
            print(theta)
            self.pose = Point()
            self.rate = rospy.Rate(1000)
                
            radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x) #二点間の角度

            if theta - radian < 0:
                vel_msg.angular.z = 0.5
                vel_msg.angular.x = 0
                vel_msg.angular.y = 0
            else:
                vel_msg.angular.z = -0.5
                vel_msg.angular.x = 0
                vel_msg.angular.y = 0
                
            #Publishing our vel_msg
            self.velocity_publisher.publish(vel_msg)

            print(theta)
            if abs(radian - theta) <=0.005: #方向が一致するまで回転
                break
            #Stopping our robot after the movement is over
        vel_msg.linear.x = 0
        vel_msg.angular.z =0
        self.velocity_publisher.publish(vel_msg) #停止
        rospy.sleep(1.5)
        while sqrt(pow((goal_pose.x - self.resp_coordinates.pose.position.x), 2) + pow((goal_pose.y - self.resp_coordinates.pose.position.y), 2)) >= distance_tolerance: #現座標と目標座標との距離が一定値以下
            vel_msg.linear.x = 0.5
            vel_msg.linear.y = 0
            vel_msg.angular.z =0
            self.velocity_publisher.publish(vel_msg) #直進
            #rospy.init_node('turtlebot_controller', anonymous=True)
            self.velocity_publisher = rospy.Publisher('r1/mobile_base/commands/velocity', Twist, queue_size=10)
            self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            pts = []
            name = []
            block = _blockListDict["r1"]
            blockName = str(block._name)
            self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
            print(self.resp_coordinates.pose.position.x,self.resp_coordinates.pose.position.y)
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()
            radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x) #二点間の角度
            if abs(radian - theta) >=0.05:
                vel_msg.linear.x = 0
                self.velocity_publisher.publish(vel_msg)
                rospy.sleep(1.0)
                while True:
                    rospy.init_node('turtlebot_controller', anonymous=True)
                    self.velocity_publisher = rospy.Publisher('r1/mobile_base/commands/velocity', Twist, queue_size=10)
                    self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                    pts = []
                    name = []
                    block = _blockListDict["r1"]
                    blockName = str(block._name)
                    self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
                    #pts = (resp_coordinates.pose.position.x, resp_coordinates.pose.position.y)
                    #print(resp_coordinates)
                    rot_q = self.resp_coordinates.pose.orientation
                    (roll,pitch,theta) = tf.transformations.euler_from_quaternion([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
                    print(theta)
                    radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x)

                    if theta - radian < 0:
                        vel_msg.angular.z = 0.5
                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0
                    else:
                        vel_msg.angular.z = -0.5
                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0               
                    #Publishing our vel_msg
                    self.velocity_publisher.publish(vel_msg)
                    if abs(radian - theta) <=0.005: #方向が一致するまで回転
                        vel_msg.angular.z =0
                        self.velocity_publisher.publish(vel_msg)
                        rospy.sleep(1)
                        break
        vel_msg.linear.x = 0
        vel_msg.angular.z =0
        self.velocity_publisher.publish(vel_msg)
        #rospy.spin()

    def r2goal(self):
        print(mydict["r2"][0])
        goal_pose = Po()
        goal_pose.x = mydict["r2"][0]
        goal_pose.y = mydict["r2"][1]
        distance_tolerance = 0.1
        vel_msg = Twist()
        while True:
            _blockListDict = {
            'r1': Block('r1', ''),
            'r2': Block('r2', ''),
            'r3': Block('r3', ''),
            'r4': Block('r4', ''),
            'r5': Block('r5', ''),
            }
            #Creating our node,publisher and subscriber
            rospy.init_node('turtlebot_controller', anonymous=True)
            self.velocity_publisher = rospy.Publisher('r2/mobile_base/commands/velocity', Twist, queue_size=10)
            self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            pts = []
            name = []
            block = _blockListDict["r2"]
            blockName = str(block._name)
            self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
            #pts = (resp_coordinates.pose.position.x, resp_coordinates.pose.position.y)
            #print(resp_coordinates)
            rot_q = self.resp_coordinates.pose.orientation
            global theta
            (roll,pitch,theta) = tf.transformations.euler_from_quaternion([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
            print(theta)
            self.pose = Point()
            self.rate = rospy.Rate(1000)
                
            radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x) #二点間の角度

            if theta - radian < 0:
                vel_msg.angular.z = 0.5
                vel_msg.angular.x = 0
                vel_msg.angular.y = 0
            else:
                vel_msg.angular.z = -0.5
                vel_msg.angular.x = 0
                vel_msg.angular.y = 0
                
            #Publishing our vel_msg
            self.velocity_publisher.publish(vel_msg)

            print(theta)
            if abs(radian - theta) <=0.005: #方向が一致するまで回転
                break
            #Stopping our robot after the movement is over
        vel_msg.linear.x = 0
        vel_msg.angular.z =0
        self.velocity_publisher.publish(vel_msg) #停止
        rospy.sleep(1.5)
        while sqrt(pow((goal_pose.x - self.resp_coordinates.pose.position.x), 2) + pow((goal_pose.y - self.resp_coordinates.pose.position.y), 2)) >= distance_tolerance: #現座標と目標座標との距離が一定値以下
            vel_msg.linear.x = 0.5
            vel_msg.linear.y = 0
            vel_msg.angular.z =0
            self.velocity_publisher.publish(vel_msg) #直進
            #rospy.init_node('turtlebot_controller', anonymous=True)
            self.velocity_publisher = rospy.Publisher('r2/mobile_base/commands/velocity', Twist, queue_size=10)
            self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            pts = []
            name = []
            block = _blockListDict["r2"]
            blockName = str(block._name)
            self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
            print(self.resp_coordinates.pose.position.x,self.resp_coordinates.pose.position.y)
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()
            radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x) #二点間の角度
            if abs(radian - theta) >=0.05:
                vel_msg.linear.x = 0
                self.velocity_publisher.publish(vel_msg)
                rospy.sleep(1.0)
                while True:
                    rospy.init_node('turtlebot_controller', anonymous=True)
                    self.velocity_publisher = rospy.Publisher('r2/mobile_base/commands/velocity', Twist, queue_size=10)
                    self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                    pts = []
                    name = []
                    block = _blockListDict["r2"]
                    blockName = str(block._name)
                    self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
                    #pts = (resp_coordinates.pose.position.x, resp_coordinates.pose.position.y)
                    #print(resp_coordinates)
                    rot_q = self.resp_coordinates.pose.orientation
                    (roll,pitch,theta) = tf.transformations.euler_from_quaternion([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
                    print(theta)
                    radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x)

                    if theta - radian < 0:
                        vel_msg.angular.z = 0.5
                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0
                    else:
                        vel_msg.angular.z = -0.5
                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0               
                    #Publishing our vel_msg
                    self.velocity_publisher.publish(vel_msg)
                    if abs(radian - theta) <=0.005: #方向が一致するまで回転
                        vel_msg.angular.z =0
                        self.velocity_publisher.publish(vel_msg)
                        rospy.sleep(1)
                        break
        vel_msg.linear.x = 0
        vel_msg.angular.z =0
        self.velocity_publisher.publish(vel_msg)
        #rospy.spin()

    def r3goal(self):
        print(mydict["r3"][0])
        goal_pose = Po()
        goal_pose.x = mydict["r3"][0]
        goal_pose.y = mydict["r3"][1]
        distance_tolerance = 0.1
        vel_msg = Twist()
        while True:
            _blockListDict = {
            'r1': Block('r1', ''),
            'r2': Block('r2', ''),
            'r3': Block('r3', ''),
            'r4': Block('r4', ''),
            'r5': Block('r5', ''),
            }
            #Creating our node,publisher and subscriber
            rospy.init_node('turtlebot_controller', anonymous=True)
            self.velocity_publisher = rospy.Publisher('r3/mobile_base/commands/velocity', Twist, queue_size=10)
            self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            pts = []
            name = []
            block = _blockListDict["r3"]
            blockName = str(block._name)
            self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
            #pts = (resp_coordinates.pose.position.x, resp_coordinates.pose.position.y)
            #print(resp_coordinates)
            rot_q = self.resp_coordinates.pose.orientation
            global theta
            (roll,pitch,theta) = tf.transformations.euler_from_quaternion([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
            print(theta)
            self.pose = Point()
            self.rate = rospy.Rate(1000)
                
            radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x) #二点間の角度

            if theta - radian < 0:
                vel_msg.angular.z = 0.5
                vel_msg.angular.x = 0
                vel_msg.angular.y = 0
            else:
                vel_msg.angular.z = -0.5
                vel_msg.angular.x = 0
                vel_msg.angular.y = 0
                
            #Publishing our vel_msg
            self.velocity_publisher.publish(vel_msg)

            print(theta)
            if abs(radian - theta) <=0.005: #方向が一致するまで回転
                break
            #Stopping our robot after the movement is over
        vel_msg.linear.x = 0
        vel_msg.angular.z =0
        self.velocity_publisher.publish(vel_msg) #停止
        rospy.sleep(1.5)
        while sqrt(pow((goal_pose.x - self.resp_coordinates.pose.position.x), 2) + pow((goal_pose.y - self.resp_coordinates.pose.position.y), 2)) >= distance_tolerance: #現座標と目標座標との距離が一定値以下
            vel_msg.linear.x = 0.5
            vel_msg.linear.y = 0
            vel_msg.angular.z =0
            self.velocity_publisher.publish(vel_msg) #直進
            #rospy.init_node('turtlebot_controller', anonymous=True)
            self.velocity_publisher = rospy.Publisher('r3/mobile_base/commands/velocity', Twist, queue_size=10)
            self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            pts = []
            name = []
            block = _blockListDict["r3"]
            blockName = str(block._name)
            self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
            print(self.resp_coordinates.pose.position.x,self.resp_coordinates.pose.position.y)
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()
            radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x) #二点間の角度
            if abs(radian - theta) >=0.05:
                vel_msg.linear.x = 0
                self.velocity_publisher.publish(vel_msg)
                rospy.sleep(1.0)
                while True:
                    rospy.init_node('turtlebot_controller', anonymous=True)
                    self.velocity_publisher = rospy.Publisher('r3/mobile_base/commands/velocity', Twist, queue_size=10)
                    self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                    pts = []
                    name = []
                    block = _blockListDict["r3"]
                    blockName = str(block._name)
                    self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
                    #pts = (resp_coordinates.pose.position.x, resp_coordinates.pose.position.y)
                    #print(resp_coordinates)
                    rot_q = self.resp_coordinates.pose.orientation
                    (roll,pitch,theta) = tf.transformations.euler_from_quaternion([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
                    print(theta)
                    radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x)

                    if theta - radian < 0:
                        vel_msg.angular.z = 0.5
                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0
                    else:
                        vel_msg.angular.z = -0.5
                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0               
                    #Publishing our vel_msg
                    self.velocity_publisher.publish(vel_msg)
                    if abs(radian - theta) <=0.005: #方向が一致するまで回転
                        vel_msg.angular.z =0
                        self.velocity_publisher.publish(vel_msg)
                        rospy.sleep(1)
                        break
        vel_msg.linear.x = 0
        vel_msg.angular.z =0
        self.velocity_publisher.publish(vel_msg)
        #rospy.spin()

    def r4goal(self):
        print(mydict["r4"][0])
        goal_pose = Po()
        goal_pose.x = mydict["r4"][0]
        goal_pose.y = mydict["r4"][1]
        distance_tolerance = 0.1
        vel_msg = Twist()
        while True:
            _blockListDict = {
            'r1': Block('r1', ''),
            'r2': Block('r2', ''),
            'r3': Block('r3', ''),
            'r4': Block('r4', ''),
            'r5': Block('r5', ''),
            }
            #Creating our node,publisher and subscriber
            rospy.init_node('turtlebot_controller', anonymous=True)
            self.velocity_publisher = rospy.Publisher('r4/mobile_base/commands/velocity', Twist, queue_size=10)
            self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            pts = []
            name = []
            block = _blockListDict["r4"]
            blockName = str(block._name)
            self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
            #pts = (resp_coordinates.pose.position.x, resp_coordinates.pose.position.y)
            #print(resp_coordinates)
            rot_q = self.resp_coordinates.pose.orientation
            global theta
            (roll,pitch,theta) = tf.transformations.euler_from_quaternion([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
            print(theta)
            self.pose = Point()
            self.rate = rospy.Rate(1000)
                
            radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x) #二点間の角度

            if theta - radian < 0:
                vel_msg.angular.z = 0.5
                vel_msg.angular.x = 0
                vel_msg.angular.y = 0
            else:
                vel_msg.angular.z = -0.5
                vel_msg.angular.x = 0
                vel_msg.angular.y = 0
                
            #Publishing our vel_msg
            self.velocity_publisher.publish(vel_msg)

            print(theta)
            if abs(radian - theta) <=0.005: #方向が一致するまで回転
                break
            #Stopping our robot after the movement is over
        vel_msg.linear.x = 0
        vel_msg.angular.z =0
        self.velocity_publisher.publish(vel_msg) #停止
        rospy.sleep(1.5)
        while sqrt(pow((goal_pose.x - self.resp_coordinates.pose.position.x), 2) + pow((goal_pose.y - self.resp_coordinates.pose.position.y), 2)) >= distance_tolerance: #現座標と目標座標との距離が一定値以下
            vel_msg.linear.x = 0.5
            vel_msg.linear.y = 0
            vel_msg.angular.z =0
            self.velocity_publisher.publish(vel_msg) #直進
            #rospy.init_node('turtlebot_controller', anonymous=True)
            self.velocity_publisher = rospy.Publisher('r4/mobile_base/commands/velocity', Twist, queue_size=10)
            self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            pts = []
            name = []
            block = _blockListDict["r4"]
            blockName = str(block._name)
            self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
            print(self.resp_coordinates.pose.position.x,self.resp_coordinates.pose.position.y)
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()
            radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x) #二点間の角度
            if abs(radian - theta) >=0.05:
                vel_msg.linear.x = 0
                self.velocity_publisher.publish(vel_msg)
                rospy.sleep(1.0)
                while True:
                    rospy.init_node('turtlebot_controller', anonymous=True)
                    self.velocity_publisher = rospy.Publisher('r4/mobile_base/commands/velocity', Twist, queue_size=10)
                    self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                    pts = []
                    name = []
                    block = _blockListDict["r4"]
                    blockName = str(block._name)
                    self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
                    #pts = (resp_coordinates.pose.position.x, resp_coordinates.pose.position.y)
                    #print(resp_coordinates)
                    rot_q = self.resp_coordinates.pose.orientation
                    (roll,pitch,theta) = tf.transformations.euler_from_quaternion([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
                    print(theta)
                    radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x)

                    if theta - radian < 0:
                        vel_msg.angular.z = 0.5
                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0
                    else:
                        vel_msg.angular.z = -0.5
                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0               
                    #Publishing our vel_msg
                    self.velocity_publisher.publish(vel_msg)
                    if abs(radian - theta) <=0.005: #方向が一致するまで回転
                        vel_msg.angular.z =0
                        self.velocity_publisher.publish(vel_msg)
                        rospy.sleep(1)
                        break
        vel_msg.linear.x = 0
        vel_msg.angular.z =0
        self.velocity_publisher.publish(vel_msg)
        #rospy.spin()

    def r5goal(self):
        print(mydict["r5"][0])
        goal_pose = Po()
        goal_pose.x = mydict["r5"][0]
        goal_pose.y = mydict["r5"][1]
        distance_tolerance = 0.1
        vel_msg = Twist()
        while True:
            _blockListDict = {
            'r1': Block('r1', ''),
            'r2': Block('r2', ''),
            'r3': Block('r3', ''),
            'r4': Block('r4', ''),
            'r5': Block('r5', ''),
            }
            #Creating our node,publisher and subscriber
            rospy.init_node('turtlebot_controller', anonymous=True)
            self.velocity_publisher = rospy.Publisher('r5/mobile_base/commands/velocity', Twist, queue_size=10)
            self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            pts = []
            name = []
            block = _blockListDict["r5"]
            blockName = str(block._name)
            self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
            #pts = (resp_coordinates.pose.position.x, resp_coordinates.pose.position.y)
            #print(resp_coordinates)
            rot_q = self.resp_coordinates.pose.orientation
            global theta
            (roll,pitch,theta) = tf.transformations.euler_from_quaternion([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
            print(theta)
            self.pose = Point()
            self.rate = rospy.Rate(1000)
                
            radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x) #二点間の角度

            if theta - radian < 0:
                vel_msg.angular.z = 0.5
                vel_msg.angular.x = 0
                vel_msg.angular.y = 0
            else:
                vel_msg.angular.z = -0.5
                vel_msg.angular.x = 0
                vel_msg.angular.y = 0
                
            #Publishing our vel_msg
            self.velocity_publisher.publish(vel_msg)

            print(theta)
            if abs(radian - theta) <=0.005: #方向が一致するまで回転
                break
            #Stopping our robot after the movement is over
        vel_msg.linear.x = 0
        vel_msg.angular.z =0
        self.velocity_publisher.publish(vel_msg) #停止
        rospy.sleep(1.5)
        while sqrt(pow((goal_pose.x - self.resp_coordinates.pose.position.x), 2) + pow((goal_pose.y - self.resp_coordinates.pose.position.y), 2)) >= distance_tolerance: #現座標と目標座標との距離が一定値以下
            vel_msg.linear.x = 0.5
            vel_msg.linear.y = 0
            vel_msg.angular.z =0
            self.velocity_publisher.publish(vel_msg) #直進
            #rospy.init_node('turtlebot_controller', anonymous=True)
            self.velocity_publisher = rospy.Publisher('r5/mobile_base/commands/velocity', Twist, queue_size=10)
            self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            pts = []
            name = []
            block = _blockListDict["r5"]
            blockName = str(block._name)
            self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
            print(self.resp_coordinates.pose.position.x,self.resp_coordinates.pose.position.y)
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()
            radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x) #二点間の角度
            if abs(radian - theta) >=0.05:
                vel_msg.linear.x = 0
                self.velocity_publisher.publish(vel_msg)
                rospy.sleep(1.0)
                while True:
                    rospy.init_node('turtlebot_controller', anonymous=True)
                    self.velocity_publisher = rospy.Publisher('r5/mobile_base/commands/velocity', Twist, queue_size=10)
                    self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                    pts = []
                    name = []
                    block = _blockListDict["r5"]
                    blockName = str(block._name)
                    self.resp_coordinates = self.model_coordinates(blockName, block._relative_entity_name)
                    #pts = (resp_coordinates.pose.position.x, resp_coordinates.pose.position.y)
                    #print(resp_coordinates)
                    rot_q = self.resp_coordinates.pose.orientation
                    (roll,pitch,theta) = tf.transformations.euler_from_quaternion([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
                    print(theta)
                    radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x -self.resp_coordinates.pose.position.x)

                    if theta - radian < 0:
                        vel_msg.angular.z = 0.5
                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0
                    else:
                        vel_msg.angular.z = -0.5
                        vel_msg.angular.x = 0
                        vel_msg.angular.y = 0               
                    #Publishing our vel_msg
                    self.velocity_publisher.publish(vel_msg)
                    if abs(radian - theta) <=0.005: #方向が一致するまで回転
                        vel_msg.angular.z =0
                        self.velocity_publisher.publish(vel_msg)
                        rospy.sleep(1)
                        break
        vel_msg.linear.x = 0
        vel_msg.angular.z =0
        self.velocity_publisher.publish(vel_msg)




if __name__ == '__main__':
    try:
        mydict = dict()
        tuto = Tutorial()
        tuto.show_gazebo_models()
        print(mydict)
        theta = None
        x = turtlebot()
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=5)
        executor.submit(x.r1goal())
        executor.submit(x.r2goal())
        executor.submit(x.r3goal())
        executor.submit(x.r4goal())
        executor.submit(x.r5goal())
    except rospy.ROSInterruptException: pass
    


