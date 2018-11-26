#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist

if __name__=="__main__":
        rospy.init_node('it')

        p=rospy.Publisher('r5/mobile_base/commands/velocity',Twist)
        twist = Twist()
        twist.linear.x = 1;
        twist.linear.y = 1;
        twist.linear.z = 0;

        twist.angular.x = 1;
        twist.angular.y = 1;
        twist.angular.z = 0.0;
        for i in range(10):
                p.publish(twist);
                rospy.sleep(0.5)
