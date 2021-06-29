"""Script for generating sequences of traffic sign detection training data. Imitates how a moving camera would capture
multiple frames of a scene where the target object is at slightly different distances from the camera in each frame.
"""
# Author: Kristian Rados

OUT_DIR     = "SGTSD_Sequences"
LABELS_FILE = "labels.txt"
import argparse
import cv2
from datetime import datetime
import numpy as np
import os
from utils import load_paths, load_files, resize, overlay


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("bg_dir", type=dir_path, help="path to background image directory")
    parser.add_argument("fg_dir", type=dir_path, help="path to foreground/sign image directory")
    return parser.parse_args()

def dir_path(path):
    """Validate an argument type as being a directory path."""
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"""{path} is not a valid path""")


def draw_square(event, x, y, flags, param):
    """Mouse callback function for drawing."""
    img = param[0]

    # Draw square to indicate selection to user
    if event == cv2.EVENT_LBUTTONDOWN:
        current_anchor_set = param[1]
        if len(current_anchor_set) <= 1:
            cv2.drawMarker(img, (x,y), (0,255,0), cv2.MARKER_SQUARE, 40, 2)
            current_anchor_set.append((x,y))  # Append anchor coordinates to anchors list
        else:
            print("Error: you have already selected both anchors. Press [r] to reset the anchors for this image.")

def select_anchor_points(bg_paths, num_bg):
    """ """  # TODO: Docstring
    anchors = []
    count = 1
    for bg in bg_paths:
        print(f"Selecting anchor points for background image {count}/{num_bg}...")
        window_name = "Select anchor points for " + bg

        img = cv2.imread(bg, cv2.IMREAD_UNCHANGED)
        current_anchor_set = []  # Set of 2 tuples, indicating far anchor and near anchor respectively
        selecting = True;

        # Set up interactive window for user anchor point selection
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, draw_square, param=[img, current_anchor_set])

        while(selecting):
            cv2.imshow(window_name, img)
            k = cv2.waitKey(1) & 0xFF

            # Confirm selection and move onto next background image
            if k == 32 or k == 13:  # Press SPACEBAR key or ENTER key
                if len(current_anchor_set) == 2:
                    anchors.append(current_anchor_set)
                    count += 1
                    selecting = False
                else:
                    print("Error: you must first select 2 anchor points.")

            # Reset the image as a means of undoing changes
            elif k == ord('r') or k == ord('R'):  # Press 'r' key or 'R' key
                current_anchor_set = []
                img = cv2.imread(bg, cv2.IMREAD_UNCHANGED)
                cv2.setMouseCallback(window_name, draw_square, param=[img, current_anchor_set])

            # Quit early to 
            elif k == 27:  # ESC key
                raise InterruptedError("quitting early")
        cv2.destroyAllWindows()
    return anchors


def main():
    args = parse_arguments()
    labels_path = os.path.join(OUT_DIR, LABELS_FILE)

    # Loading argument-specified directories
    bg_paths = load_paths(args.bg_dir)
    fg_paths = load_paths(args.fg_dir)
    num_bg = len(bg_paths)
    num_fg = len(fg_paths)
    print(f"Found {num_bg} background images.")
    print(f"Found {num_fg} foreground images.\n")
    if num_bg == 0 or num_fg == 0:
        print("Error: each input directory must have at least 1 image")
        return

    # Directory structure setup
    if (not os.path.exists(OUT_DIR)):
        out_dir = OUT_DIR
    else:
        timestamp = datetime.now().strftime("_%d%m%Y_%H%M%S")
        out_dir = OUT_DIR + timestamp
    os.mkdir(out_dir)

    # Get user to indicate anchor points in the image
    try:
        anchors = select_anchor_points(bg_paths, num_bg)  # TODO: Extend to allow >1 AP sets per bg image
    except InterruptedError as e:
        print("Error:", str(e))
        return
    
    print(anchors) #

if __name__ == "__main__":
    main()