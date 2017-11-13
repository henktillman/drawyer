# import the necessary packages
import numpy as np
# import argparse
from skimage.filters import threshold_adaptive
import cv2, pdb
 
def order_points(pts):
  # Initialize a list of coordinates that will be ordered
  # such that the first entry in the list is the top-left,
  # the second entry is the top-right, the third is the
  # bottom-right, and the fourth is the bottom-left.
  rect = np.zeros((4, 2), dtype = "float32")
 
  # The top-left point will have the smallest sum, whereas
  # the bottom-right point will have the largest sum.
  s = pts.sum(axis = 1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]
 
  # Now, compute the difference between the points. the
  # top-right point will have the smallest difference,
  # whereas the bottom-left will have the largest difference.
  diff = np.diff(pts, axis = 1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]
 
  # Return the ordered coordinates.
  return rect


# Given an image and four points, transforms the image so that the rectangle
# defined by those points fills the view.
def four_point_transform(image, pts):
  # Obtain a consistent order of the points and unpack them
  # individually.
  rect = order_points(pts)
  (tl, tr, br, bl) = rect
 
  # Compute the width of the new image, which will be the
  # maximum distance between bottom-right and bottom-left
  # x-coordiates or the top-right and top-left x-coordinates.
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
 
  # Compute the height of the new image, which will be the
  # maximum distance between the top-right and bottom-right
  # y-coordinates or the top-left and bottom-left y-coordinates.
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))
 
  # Now that we have the dimensions of the new image, construct
  # the set of destination points to obtain a "birds eye view",
  # (i.e. top-down view) of the image, again specifying points
  # in the top-left, top-right, bottom-right, and bottom-left
  # order.
  dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
 
  # compute the perspective transform matrix and then apply it
  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
  # return the warped image
  return warped

# Given an image and a height, resizes the image to have that height and keep it's
# current width/height ratio.
def resize(image, height):
  ratio = float(height) / image.shape[0]
  dim = (int(image.shape[1] * ratio), int(height))
  return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


image = cv2.imread("./test.jpg")
ratio = image.shape[0] / 700.0
orig = image.copy()
image = resize(image, 700)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
 
# loop over the contours
for c in cnts:
  # approximate the contour
  peri = cv2.arcLength(c, True)
  approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
  # if our approximated contour has four points, then we
  # can assume that we have found our screen
  if len(approx) == 4:
    screenCnt = approx
    break
 
# apply the four point transform to obtain a top-down
# view of the original image
# pdb.set_trace()
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = threshold_adaptive(warped, 251, offset = 10)
warped = warped.astype("uint8") * 255
 
# show the original and scanned images
cv2.imshow("Original", resize(orig, 700))
cv2.imshow("Scanned", resize(warped, 700))
cv2.waitKey(0)
cv2.destroyAllWindows()