"""
Extended from create_sequences.py

Script for generating sequences of traffic sign detection training data. Imitates how a moving camera would capture
multiple frames of a scene where the target object is at slightly different distances from the camera in each frame.

The distance of the near and far anchor points are passed as command line arguments, and a set of foregrounds, with
evenly distributed distances over the near and far anchor distances, are produced via perspective projection. 

The camera's focal length or field of view is also passed as a command line argument to apply perspective projection.
"""

# Author: Kristian Rados, Allen Antony

OUT_DIR         = "SGTS_Sequences"
LABELS_FILE     = "labels.txt"
MIN_ANCHOR_SIZE = 10
MAX_ANCHOR_SIZE = 200
FOVY = 60

import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "../../"))

import argparse
import cv2
from datetime import datetime
import numpy as np
import math
from pathlib import Path
from signbreaker.utils import load_paths, overlay


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("bg_dir", type=Path, help="path to background image directory")
    parser.add_argument("fg_dir", type=Path, help="path to foreground/sign image directory")
    parser.add_argument("-n", "--num_frames", type=int, help="number of frames generated for each sequence", default=8)
    parser.add_argument("-d1", "--min_dist", type=int, help="minimum distance of sign from camera", default=4)
    parser.add_argument("-d2", "--max_dist", type=int, help="maximum distance of sign from camera", default=20)
    return parser.parse_args()


class Anchor(object):
    """
    A class that stores OpenCV pixel coordinates of the top left corner of the sign, 
    as well as the size of the sign in pixels.
    """
    def __init__(self, bg_size, NDC_x, NDC_y, x_size, y_size, sign_z):
        height, width, _ = bg_size
        x_size, y_size = x_size / 2, y_size / 2
        
        # Converting from normalized device coordinates to 0-1 range
        NDC_x = (NDC_x + 1) / 2
        NDC_x = min(max(NDC_x, 0), 1 - x_size)
        
        # Converting from normalized device coordinates to 0-1 range and shifting
        # vanishing point vertically
        NDC_y = 1 - ((NDC_y + 0.45) / 1.45)
        NDC_y = min(max(NDC_y, 0), 1 - y_size)
        
        # Converting from 0-1 range to pixel coordinates
        self.size = int(x_size * width)
        self.screen_x = int(NDC_x * width)
        self.screen_y = int(NDC_y * height)
        self.distance = -1 * sign_z
        
    def __str__(self):
        return f"Anchor: {self.screen_x}, {self.screen_y}, {self.size}"
      
        
class SignObject(object):
    """
    A class that stores the world coordinates of a sign object, applying perspective projection via
    the method 'perspective_transform'
    """
    def __init__(self, x, y, z, size):
        # Top left corner of sign
        self.x1 = x 
        self.y1 = y 
        # Bottom right corner of sign
        self.x2 = x + size
        self.y2 = y - size
        # Virtual distance from camera to sign
        self.z = z
        
    def perspective_transform(self, bg_size, proj_matrix):
        """
        Applies perspective projection to the sign object to produce normalised device coordinates (NDC), 
        and initialises and returns an anchor object using these new coordinates.
        """
        # Scale the sign coordinates by the perspective projection matrix
        x1, y1, _, w = proj_matrix.dot(np.array([self.x1, self.y1, self.z, 1]))
        x2, y2, _, w = proj_matrix.dot(np.array([self.x2, self.y2, self.z, 1]))
        
        # Transform to normalized device coordinates
        x1f, y1f, x2f, y2f = x1/w, y1/w, x2/w, y2/w
        
        x_size = abs(x2f - x1f)
        y_size = abs(y2f - y1f)
        return Anchor(bg_size, NDC_x=x1f, NDC_y=y1f, x_size=x_size, y_size=y_size, sign_z=self.z)
    
    
def create_perspective(fovy, aspect, near, far):
    """[summary]
    Creates a perspective projection matrix using vertical fov, aspect ratio, and near and
    far clipping distances. 
    Code adapted from http://learnwebgl.brown37.net/08_projections/projections_perspective.html
    """
    if (fovy <= 0 or fovy >= 180 or aspect <= 0 or near >= far or near <= 0):
        raise ValueError('Invalid parameters to createPerspective')
    half_fovy = math.radians(fovy) / 2
    
    top = near * math.tan(half_fovy)
    bottom = -top
    right = top * aspect
    left = -right
    return create_frustrum(left, right, bottom, top, near, far)


def create_frustrum(left, right, bottom, top, near, far):
    """[summary]
    Creates a perspective projection matrix using the left, right, bottom, top, near and far
    bounds of the near clipping plane.
    Code adapted from http://learnwebgl.brown37.net/08_projections/projections_perspective.html
    """
    if (left == right or bottom == top or near == far):
        raise ValueError('Invalid parameters to createFrustrum')
    if (near <= 0 or far <= 0):
        raise ValueError('Near and far must be positive')
    
    perspective_matrix =  np.array([[near, 0, 0, 0],
                                    [0, near, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, -1, 0]], 
                                    dtype=np.float32) 
    NDC_matrix =  np.array([[2/(right-left), 0, 0, 0],
                            [0, 2/(top-bottom), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]],
                            dtype=np.float32)
    return np.dot(NDC_matrix, perspective_matrix)
    

def produce_anchors(bg_size, size, x, y, min_dist, max_dist, num_frames):
    """[summary]
    Generates a list of anchor objects, by applying perspective projection on the constant
    world coordinates, depending on virtual distance from the camera. Using np.linspace, the 
    distances are evenly distributed over min_dist to max_dist to create the anchor objects.
    """
    anchors = []
    height, width, _ = bg_size
    aspect_ratio = width / height
    proj_matrix = create_perspective(FOVY, aspect_ratio, NEAR_CLIPPING_PLANE_DIST, FAR_CLIPPING_PLANE_DIST)
    
    for dist in np.linspace(max_dist, min_dist, num=num_frames, endpoint=True):
        sign_z = -1 * dist      # Projection matrix assumes negative z values in front of camera
        sign_near = SignObject(x, y, z=sign_z, size=size)
        anchor = sign_near.perspective_transform(bg_size, proj_matrix)
        anchors.append(anchor)
    return anchors


SIGN_COORDS = {'x':1.5, 'y':1, 'size':0.5}      # World coordinates for rendered sign objects
NEAR_CLIPPING_PLANE_DIST = 2
FAR_CLIPPING_PLANE_DIST = 50


def create_sequence(bg_img, fg_img, bg_name, fg_name, sequence_id, out_dir, min_dist=4, max_dist=20, num_frames=8):
    anchors = produce_anchors(bg_img.shape, SIGN_COORDS['size'], SIGN_COORDS['x'], SIGN_COORDS['y'], 
                              min_dist, max_dist, num_frames)
    bounding_boxes = []
    image_paths = []
    
    for frame, anchor in enumerate(anchors):
        save_path = f"{out_dir}/{sequence_id + frame}-{bg_name}-{fg_name}-{frame}.jpg"
        scaled_fg_img = cv2.resize(fg_img, (anchor.size, anchor.size))
        new_img = overlay(scaled_fg_img, bg_img, anchor.screen_x, anchor.screen_y)
        cv2.imwrite(save_path, new_img)
        
        bounding_boxes.append([anchor.screen_x, anchor.screen_y, anchor.size, anchor.size])
        image_paths.append(save_path)
    return (image_paths, bounding_boxes)


if __name__ == '__main__':
    args = parse_arguments()
    
    # Loading argument-specified directories
    bg_paths = load_paths(args.bg_dir)
    fg_paths = load_paths(args.fg_dir)
    
    print(f"Found {len(bg_paths)} background images.")
    print(f"Found {len(fg_paths)} foreground images.\n")
    
    if os.listdir(args.bg_dir) == [] or os.listdir(args.fg_dir) == []:
        raise ValueError("Error: each input directory must have at least 1 image")
    
    # Directory structure setup
    if (not os.path.exists(OUT_DIR)):
        out_dir = OUT_DIR
    else:
        timestamp = datetime.now().strftime("_%d%m%Y_%H%M%S")
        out_dir = OUT_DIR + timestamp
    os.mkdir(out_dir)

    # Generate sequences by overlaying foregrounds over backgrounds according to anchor point data
    for bg_path in bg_paths:
        bg_img = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
        for fg_path in fg_paths:
            fg_img = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
            # Path without extension
            save_path = f'{out_dir}/{Path(bg_path).stem}-{Path(fg_path).stem}'
            create_sequence(bg_img, fg_img, fg_path, save_path, 
                            args.min_dist, args.max_dist, args.num_frames)
    
    