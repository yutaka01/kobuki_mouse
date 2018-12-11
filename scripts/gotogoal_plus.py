#!/usr/bin/env python
import rospy
import tf
from geometry_msgs.msg  import Twist,Point,Quaternion
from turtlesim.msg import Pose
from math import pow,atan2,sqrt
from gazebo_msgs.srv import GetModelState
import numpy as np
from numpy import linalg,cos

class Block:
    def __init__(self, name, relative_entity_name):
        self._name = name
        self._relative_entity_name = relative_entity_name

class turtlebot():
        
    
    #Callback function implementing the pose value received
    def callback(self, data):
        self.pose = data
        self.pose.x = round(self.pose.x, 4)
        self.pose.y = round(self.pose.y, 4)

    def get_distance(self, goal_x, goal_y):
        distance = sqrt(pow((goal_x - self.pose.x), 2) + pow((goal_y - self.pose.y), 2))
        return distance

    def move2goal(self):
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
            goal_pose = Point()
            goal_pose.x = 5#input("Set your x goal:")
            goal_pose.y = 5#input("Set your y goal:")
            distance_tolerance = 0.5
            vel_msg = Twist()
                

        #Porportional Controller
        #linear velocity in the x-axis:
            radian = atan2(goal_pose.y - self.resp_coordinates.pose.position.y, goal_pose.x - self.resp_coordinates.pose.position.x)

 
            vel_msg.linear.x = 0* sqrt(pow((goal_pose.x - self.resp_coordinates.pose.position.x), 2) + pow((goal_pose.y - self.resp_coordinates.pose.position.y), 2))
            vel_msg.linear.y = 0
            vel_msg.linear.z = 0

        #angular velocity in the z-axis:

            vel_msg.angular.z = 1
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            #Publishing our vel_msg
            self.velocity_publisher.publish(vel_msg)

            print(theta)
            print(cos(theta))
            if abs(radian - theta) <=0.005:
                break
            #Stopping our robot after the movement is over
        vel_msg.linear.x = 0
        vel_msg.angular.z =0
        self.velocity_publisher.publish(vel_msg)
        rospy.sleep(1)
        #rospy.spin()
        num = 0
        #while sqrt(pow((goal_pose.x - self.resp_coordinates.pose.position.x), 2) + pow((goal_pose.y - self.resp_coordinates.pose.position.y), 2)) >= distance_tolerance:
        while True:
            num += 1
            vel_msg.linear.x = 1
            self.velocity_publisher.publish(vel_msg)
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
            if num == 1000:
                break
        vel_msg.linear.x = 0
        vel_msg.angular.z =0
        self.velocity_publisher.publish(vel_msg)
        rospy.spin()

if __name__ == '__main__':
    try:
        #Testing our function
        theta = None
        x = turtlebot()
        x.move2goal()

    except rospy.ROSInterruptException: pass
