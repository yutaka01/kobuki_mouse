#! /usr/bin/env python

from gazebo_msgs.srv import GetModelState
import rospy
if __name__ == '__main__':
    rospy.wait_for_service('/gazebo/get_model_state')
    model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    rospy.loginfo(model_state.position.x)
    print "Service call failed"
