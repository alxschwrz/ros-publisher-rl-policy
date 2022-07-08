#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray


def callback(data):
    pass

def listener():

    rospy.init_node('action_listener', anonymous=True)

    rospy.Subscriber('chatter', Float32MultiArray, callback)

    rospy.spin()


if __name__ == '__main__':
    listener()
