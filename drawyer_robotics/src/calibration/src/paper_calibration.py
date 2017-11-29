import numpy as np
import pickle
from sensing import get_marker_pos

def get_paper_corners():
    raw_input("Please move marker to top left corner of paper, then hit Enter. ")
    top_left = get_marker_pos()
    print("Got top left coordinates: {}".format(top_left))
    
    raw_input("\nPlease move marker to bottom right corner of paper, then hit Enter. ")
    bottom_right = get_marker_pos()
    print("Got bottom right coordinates: {}".format(bottom_right))

    if top_left[0] >= bottom_right[0]:
        raise Exception("ERROR: top_left x-coord must be less than bottom_right x-coord.")
    if top_left[1] <= bottom_right[1]:
        raise Exception("ERROR: top_left y-coord must be greater than bottom_right y-coord.")
    if abs(top_left[2] - bottom_right[2]) > 0.05:
        raise Exception("ERROR: top_left z-coord must be close (<5cm) to bottom_right z-coord.")

    pickle.dump((top_left, bottom_right), "coordinates.pickle")

    return (top_left, bottom_right)

if __name__ == "__main__":
    print(get_paper_corners())
