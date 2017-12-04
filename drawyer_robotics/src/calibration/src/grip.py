from intera_interface import gripper as robot_gripper
import rospy

def main():
	rospy.init_node('moveit_node')
	right_gripper = robot_gripper.Gripper('right')
	print('Opening...')
	right_gripper.open()
	rospy.sleep(1.0)
	
	print('Calibrating...')
	right_gripper.calibrate()
	rospy.sleep(2.0)

	print('Opening...')
	right_gripper.open()
	rospy.sleep(1.0)

	raw_input('Insert the block and hit <Enter> when done. ')

	print('Closing...')
	right_gripper.close()
	rospy.sleep(1.0)
	print('Done!')

if __name__ == '__main__':
	main()