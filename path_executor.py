#!/usr/bin/env python
import sys
import rospy
import moveit_commander
from moveit_msgs.msg import OrientationConstraint, Constraints
from geometry_msgs.msg import PoseStamped
from baxter_interface import gripper as robot_gripper
import pickle, pdb



def move_to_point(arm, point):
  rospy.sleep(1.0)
  goal = PoseStamped()
  goal.header.frame_id = "base"

  #x, y, and z position
  goal.pose.position.x = point[0]
  goal.pose.position.y = point[1] 
  goal.pose.position.z = point[2]
  
  #Orientation as a quaternion
  goal.pose.orientation.x = 0.0
  goal.pose.orientation.y = -1.0
  goal.pose.orientation.z = 0.0
  goal.pose.orientation.w = 0.0

  #Set the goal state to the pose you just defined
  arm.set_pose_target(goal)
  #Set the start state for the right arm
  arm.set_start_state_to_current_state()

  # #Create a path constraint for the arm
  # #UNCOMMENT TO ENABLE ORIENTATION CONSTRAINTS
  orien_const = OrientationConstraint()

  # change to left if using baxter's left arm
  orien_const.link_name = "right_gripper";
  orien_const.header.frame_id = "base";
  orien_const.orientation.y = -1.0;
  orien_const.absolute_x_axis_tolerance = 0.1;
  orien_const.absolute_y_axis_tolerance = 0.1;
  orien_const.absolute_z_axis_tolerance = 0.1;
  orien_const.weight = 1.0;
  consts = Constraints()
  consts.orientation_constraints = [orien_const]
  arm.set_path_constraints(consts)

  #Plan a path
  plan = arm.plan()

  #Execute the plan
  raw_input('Press <Enter> to move the arm to the next pose: ')
  arm.execute(plan)


def main():
  ##########################################################################################################
  #Calibration data ----------------------------------------------------------------------------------------
  ##########################################################################################################
  # Load the top left and bottom right rectangle coordinates.
  with open('calibration.pickle', 'rb') as handle:
    calibration_coords = pickle.load(handle)
  top_left = calibration_coords[0]
  bottom_left = calibration_coords[1]
  # This constant represents how far above the paper (in z coordinates) the end effector should be when
  # is is not drawing a curve.
  gap = 0.1


  ##########################################################################################################
  #Robot arm initialization and calibration ----------------------------------------------------------------
  ##########################################################################################################
  #Initialize moveit_commander
  moveit_commander.roscpp_initialize(sys.argv)

  #Start a node
  rospy.init_node('moveit_node')

  #Initialize both arms
  robot = moveit_commander.RobotCommander()
  scene = moveit_commander.PlanningSceneInterface()
  # Uncomment if you need to use baxter's left arm. Sawyer's arm is right by default.
  # left_arm = moveit_commander.MoveGroupCommander('left_arm')

  right_arm = moveit_commander.MoveGroupCommander('right_arm')

  # left_arm.set_planner_id('RRTConnectkConfigDefault')
  # left_arm.set_planning_time(10)
  right_arm.set_planner_id('RRTConnectkConfigDefault')
  right_arm.set_planning_time(10)
  right_gripper = robot_gripper.Gripper('right')
  print('Calibrating...')
  right_gripper.calibrate()
  rospy.sleep(2.0)

  ##########################################################################################################
  #Direct the arm to a neutral position above the paper ----------------------------------------------------
  ##########################################################################################################
  goal_1 = PoseStamped()
  goal_1.header.frame_id = "base"

  #x, y, and z position
  goal_1.pose.position.x = top_left[0]
  goal_1.pose.position.y = top_left[1]
  goal_1.pose.position.z = top_left[1] + 0.3 # clearly raise it above the paper
  
  #Orientation as a quaternion
  goal_1.pose.orientation.x = 0.0
  goal_1.pose.orientation.y = -1.0
  goal_1.pose.orientation.z = 0.0
  goal_1.pose.orientation.w = 0.0

  #Set the goal state to the pose you just defined
  right_arm.set_pose_target(goal_1)

  #Set the start state for the right arm
  right_arm.set_start_state_to_current_state()

  #Plan a path
  right_plan = right_arm.plan()

  #Execute the plan
  raw_input('Press <Enter> to move the arm to the starting position')
  right_arm.execute(right_plan)

  ##########################################################################################################
  #Give the robot the gripper (comment if already done) ----------------------------------------------------
  ##########################################################################################################

  #Open the right gripper
  print('Opening...')
  right_gripper.open()
  rospy.sleep(1.0)

  raw_input('Place the block in the gripper and then press <Enter> to grip the block...')

  #Close the right gripper
  print('Closing...')
  right_gripper.close()
  rospy.sleep(1.0)
  print('Done!')




if __name__ == '__main__':
  main()
