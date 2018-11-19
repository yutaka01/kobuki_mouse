#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import ModelStates

def callback(data):
    rospy.loginfo(data)
def listener():
    rospy.init_node('state_subscribe',anonymous=True)
    rospy.Subscriber('gazebo/model_states',ModelStates,callback)
    rospy.spin()
if __name__=="__main__":
    listener()
