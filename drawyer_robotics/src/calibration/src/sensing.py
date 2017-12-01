#!/usr/bin/env python
import rospy
import tf2_ros
import numpy as np

def get_marker_pos(marker_length=0.15):
    rospy.init_node("tf_listener")

    buf = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buf)

    rospy.sleep(1.0)

    try:
        tform = buf.lookup_transform('reference/right_gripper', 'reference/base', rospy.Time(0), timeout=rospy.Duration(4))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        return e

    translation = tform.transform.translation
    translation = np.array([translation.x, translation.y, translation.z])

    q = tform.transform.rotation
    q = np.array([q.x, q.y, q.z, q.w])
    q *= np.sqrt(2)
    q = np.outer(q, q)
    rotation = np.array([[1 - q[2,2] - q[3,3], q[1,2] - q[3,0], q[1,3] + q[2,0]],
                        [q[1,2] + q[3,0], 1 - q[1,1] - q[3,3], q[2,3] - q[1,0]],
                        [q[1,3] - q[2,0], q[2,3] + q[1,0], 1 - q[1,1] - q[2,2]]])

    wrist_coords = np.concatenate(
                    (np.concatenate(
                        (rotation, np.expand_dims(translation, axis=1)),
                        axis=1),
                    np.array([[0, 0, 0, 1]])),
                    axis=0)

    marker_transform = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, -marker_length],
                                [0, 0, 0, 1]])

    marker_coords = np.matmul(wrist_coords, marker_transform)    

    return marker_coords[:3, 3]


if __name__ == "__main__":
    print(get_marker_pos())
