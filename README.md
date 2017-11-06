# BioE/EECS C106A - Final Project - Drawyer 
## Contact Information
- Stella Seo      3031829230      seo.jysk@berkeley.edu 
- Alvin Zhang     3031881529      alvinz@berkeley.edu
- Henk Tillman    3031790711      henk@berkeley.edu

## Abstract
Creating an interpretable visual representation of the real world is an essential task for robots if they are to be used in our everyday lives and communicate with humans. In our project, we will enable Baxter to recreate sketch-style drawings of objects and images. In addition to making bad caricatures, this project also incorporates image processing, computer vision, route planning, and actuation in creating the final drawing. We have additionally set stretch goals involving grasping, more advanced image processing, and style transfer (art!) via machine learning.

## Project Description
Our project will have three primary phases: image processing/CV (sensing), route planning, and actuation.

In the first phase (image processing/CV), we will hold up our desired line drawing and have Baxter take a picture of it using a webcam or a built-in camera. Baxter will convert the image to black and white, use adaptive thresholding to more clearly delineate the lines, and run the Canny edge detection algorithm in order to generate a series of points which define the lines on the page. 

In the second phase, we will use the results from the first phase to plan a path for Baxter’s arm to travel. The detected lines will be discretized into a series of points that Baxter’s arm must travel between. It will also determine whether Baxter should be drawing a line when travelling between two points. Constructing the optimal route, so that Baxter spends as little time as possible drawing, may be viewed as a modification of the Traveling Salesman Problem.

In the third phase, Baxter executes the drawing. A pen holder that he grasps in his pincer ensures that the pen remains parallel to the Z-axis of his tool frame (in other words, perpendicular to the paper). After moving his arm to the initial position on a piece of paper taped to a table in front of him, Baxter recreates the drawing by following the plan from phase II, keeping the pen at a right angle to the page and the tool frame a certain distance away from the table depending on if he should be drawing a line or not.

Following completion of our project, we will expect Baxter to be able to copy simple line drawings with near-perfect accuracy, and to perform well with noise and more complicated drawings. Please refer to the “Assessment” section for more details regarding the final product.

As far as we are aware, this is the first time that this project has been attempted for this class. A similar project being pursued is a “connect-the-dots” program, and other similar projects have certainly been done before. Nevertheless, we feel that this project is interesting due to our stretch goals that would greatly increase the expressiveness of the works produced by Baxter, including style transfer and varying level of detail, stroke width, and color.

## Tasks
- Pen-holder. We will create a pen-holder, likely with a block of Styrofoam, which is sturdy and which Baxter can grasp robustly. If the Styrofoam proves ineffective, we will try 3D-printing a holder.
- Image Processing. The parts below will be implemented as ROS nodes offering services.
- Taking the Photo. We will use a webcam or Baxter’s wrist camera to capture an image. 
- Image Processing. We will use reference points to calculate the homography between the webcam image and the desired plane Baxter is to sketch. We may also try denoising and style transfer techniques on the image at this point.
- Canny Edge Detection. We will apply the Canny edge detector to the processed image to get the edges. We may experiment with the gradient and nms thresholds at this point to see the effect on the final image.
- “Blob” Detection. In order to enumerate the contours produced by Canny edge detection, we will simplify our modified image down into an array of pixels. We can generate a corresponding 2D array that has -1’s in all of the spots that the Canny Edge Detector finds an edge. We start by scanning through these pixels and marking the ones between “edges,” where the edges must be on opposite sides vertically, horizontally, or diagonally. On our 2D matrix, we change the corresponding pixel’s value to 1. After running through all the pixels, we can run through our completed array and mark separate “blobs,” which in our case are lines of pixels that are adjacent horizontally, vertically, or diagonally to each other. 
- Route Planning.  The parts below will also be implemented as ROS nodes offering services.
- Preprocessing. Given a set of contours, we will determine the path needed to be taken by Baxter as a series of short line segments approximating the detected edges.
- Shortest Path. Given a set of line segments that Baxter is required to traverse, we will determine an order of traversal which approximately minimizes the time Baxter spends moving between line segments.
- Execution of Route. We will use MoveIt, as in Lab 5, to solve the inverse kinematics problem to get Baxter to draw along the desired line segments.
- Inverse Kinematics. We will submit a GetPositionIKRequest with appropriate path constraints (constant Z-position and wrist pitch/roll) to compute inverse kinematics. We will then forward this result to the MoveGroupCommander node to get Baxter to move along the desired line segments.

## Some stretch goals include:
- Placing the pen holder on the table and having Bexter determine a grasp before beginning to draw.
- Using rudimentary color detection on the image and having Baxter/Sawyer recreate this color by exchanging markers at certain points in the program.
- Implementing stroke thickness detection, and having the robot arm recreate it by modifying the pressure on the pen as it draws a line.
- Style transfer: using machine learning to learn the style of a certain artist, apply that style to a given drawing, and redraw the given picture with the new style.
- Varying the level of detail that Baxter uses when drawing sketches.
