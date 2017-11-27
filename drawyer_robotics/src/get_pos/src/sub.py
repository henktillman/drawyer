#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from tf2 import TFListener

def callback(message):
    print("{}".format(message.name))

#Define the method which contains the node's main functionality
def listener():
    rospy.init_node("joint_state_listener", anonymous=True)
    #rospy.Subscriber("/tf", TFMessage, callback)
    rospy.Subscriber("/robot/joint_states", JointState, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
