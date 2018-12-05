#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist

if __name__=="__main__":
        rospy.init_node('it')

        p=rospy.Publisher('r1/mobile_base/commands/velocity',Twist)
        twist = Twist()
        twist.linear.x = 0;
        twist.linear.y = 0;
        twist.linear.z = 0;

        twist.angular.x = 0;
        twist.angular.y = 0;
        twist.angular.z = 1;
        for i in range(10):
                p.publish(twist);
                rospy.sleep(0.5)
