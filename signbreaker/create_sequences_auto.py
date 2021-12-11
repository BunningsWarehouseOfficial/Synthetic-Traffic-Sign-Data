"""
Extended from create_sequences.py

Script for generating sequences of traffic sign detection training data. Imitates how a moving camera would capture
multiple frames of a scene where the target object is at slightly different distances from the camera in each frame.

The distance of the near and far anchor points are passed as command line arguments, with the anchors' coordinates
and size being scaled by perspective projection, and intermediate frames being produced via linear interpolation. 
The camera's focal length or field of view is also passed as a command line argument to apply perspective projection.
"""
# Author: Kristian Rados
# Extended by: Allen Antony

OUT_DIR         = "SGTS_Sequences"
LABELS_FILE     = "labels.txt"
MIN_ANCHOR_SIZE = 10
MAX_ANCHOR_SIZE = 200

import argparse
import cv2
from datetime import datetime
import numpy as np
import ntpath
import os
import math
from pathlib import Path
from utils import load_paths, load_files, resize, overlay


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("bg_dir", type=dir_path, help="path to background image directory")
    parser.add_argument("fg_dir", type=dir_path, help="path to foreground/sign image directory")
    parser.add_argument("-n", "--num_frames", type=int, help="number of frames generated for each sequence", default=8)
    return parser.parse_args()


class Anchor(object):
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
      
        
class SignObject(object):
    def __init__(self, centre_x, centre_y, z, size):
        # Top left corner of sign
        self.x1 = centre_x - size / 2
        self.y1 = centre_x + size / 2
        # Bottom right corner of sign
        self.x2 = centre_y + size / 2
        self.y2 = centre_y - size / 2
        # Virtual distance from camera to sign
        self.z = z
        
    def perspective_transform(self, proj_matrix):
        """
        Applies perspective projection to the sign object, and returns the coordinates of the top-left and bottom-right
        corners of the sign object in the image.
        """
        # Scale the sign coordinates by the perspective projection matrix
        x1, y1, _, w = proj_matrix.dot(np.array([self.x1, self.y1, self.z, 1]))
        x2, y2, _, w = proj_matrix.dot(np.array([self.x2, self.y2, self.z, 1]))
        x1f, y1f, x2f, y2f = x1/w, y1/w, x2/w, y2/w
        centre_x = (x1f + x2f) / 2
        centre_y = (y1f + y2f) / 2
        scaled_size = abs(x2f - x1f)
        return Anchor(x=centre_x, y=centre_y, size=scaled_size) 


def dir_path(path):
    """Validate an argument type as being a directory path."""
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"""{path} is not a valid path""")
    
    
def create_perspective(fovy, aspect, near, far):
    if (fovy <= 0 or fovy >= 180 or aspect <= 0 or near >= far or near <= 0):
        raise ValueError('Invalid parameters to createPerspective')
    half_fovy = math.radians(fovy) / 2
    
    top = near * math.tan(half_fovy)
    bottom = -top
    right = top * aspect
    left = -right

    return create_frustrum(left, right, bottom, top, near, far)


def create_frustrum(left, right, bottom, top, near, far):
    if (left == right or bottom == top or near == far):
        raise ValueError('Invalid parameters to createFrustrum')
    if (near <= 0 or far <= 0):
        raise ValueError('Near and far must be positive')
    
    sx = 2 * near / (right - left)
    sy = 2 * near / (top - bottom)
    c2 = -1 * (far + near) / (far - near)
    c1 = 2 * near * far / (near - far)
    tx = -near * (left + right) / (right - left)
    ty = -near * (bottom + top) / (top - bottom)
    
    return np.array([[    sx, 0, 0, tx    ],
                    [     0, sy, 0, ty    ],
                    [     0, 0, c2, c1    ],
                    [     0, 0, -1, 0     ]], 
                 dtype=np.float32)  
    

def produce_anchors(bg_size, fovy=45, fovx=60, size=1, centre_x=1.5, centre_y=1, sign_near_dist=6, sign_far_dist=20):
    anchors = {}
    aspect_ratio = bg_size['width'] / bg_size['height']
    if not fovy:
        fovy = 2 * math.atan(math.tan(math.radians(fovx) / 2) * aspect_ratio)
    proj_matrix = create_perspective(fovy, aspect_ratio, 0.1, 50)
    
    # Projection matrix assumes negative z values in front of camera
    sign_near_z = -1 * sign_near_dist
    sign_far_z = -1 * sign_far_dist
    
    sign_near = SignObject(centre_x, centre_y, z=sign_near_z, size=size)
    sign_far = SignObject(centre_x, centre_y, z=sign_far_z, size=size)
    
    anchors['near'] = sign_near.perspective_transform(proj_matrix)
    anchors['far'] = sign_far.perspective_transform(proj_matrix)
    return anchors
    
produce_anchors(bg_size={'width':640, 'height':480}, fovy=45, fovx=60, size=1.2)

def interpolate_frames(bg_path, fg_path, num_frames, anchors):
    bg_name = Path(bg_path).stem
    fg_name = Path(fg_path).stem
    
    seq_ratio = 1 / (num_frames - 1)     
    near_anchor = anchors['near']
    far_anchor = anchors['far']

    x_diff = near_anchor.x - far_anchor.x 
    y_diff = near_anchor.y - far_anchor.y
    size_diff = near_anchor.size - far_anchor.size

    for frame in range(num_frames):
        diff_ratio = frame * seq_ratio
        size = int(near_anchor.size + (diff_ratio * size_diff))
        x = int(near_anchor.x + (diff_ratio * x_diff))
        y = int(near_anchor.y + (diff_ratio * y_diff))

        interpolated_fg = cv2.resize(cv2.imread(fg_path), (size, size))
        img_new = overlay(interpolated_fg, cv2.imread(bg_path), x, y)
        
        img_new_path = os.path.join(OUT_DIR, bg_name + "_" + fg_name + "_" + str(frame) + ".jpg")
        cv2.imwrite(img_new_path, img_new)
    

def main():
    args = parse_arguments()

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

    anchors = produce_anchors()
    print("Anchors (x, y, size):\n" + str(anchors))

    # Generate sequences by overlaying foregrounds over backgrounds according to anchor point data
    for bg_path in bg_paths:
        for fg_path in fg_paths:
            interpolate_frames(bg_path, fg_path, args.num_frames, anchors)


if __name__ == "__main__":
    main()