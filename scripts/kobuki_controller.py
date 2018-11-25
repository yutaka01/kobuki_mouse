#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist

if __name__=="__main__":
        rospy.init_node('it')

        p=rospy.Publisher('r2/mobile_base/commands/velocity',Twist)
        twist = Twist()
        twist.linear.x = -0.5;
        twist.linear.y = 0.0;
        twist.linear.z = 0.0;

        twist.angular.x = 0.0;
        twist.angular.y = 0.0;
        twist.angular.z = 0.0;
        for i in range(10):
                p.publish(twist);
                rospy.sleep(0.5)
