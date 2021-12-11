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

NEAR_CLIPPING_PLANE_DIST = 2
FAR_CLIPPING_PLANE_DIST = 50
FOVY = 45
FOVX = 60


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("bg_dir", type=dir_path, help="path to background image directory")
    parser.add_argument("fg_dir", type=dir_path, help="path to foreground/sign image directory")
    parser.add_argument("-n", "--num_frames", type=int, help="number of frames generated for each sequence", default=8)
    return parser.parse_args()


class Anchor(object):
    def __init__(self, bg_size, NDC_x, NDC_y, x_size):
        height, width, _ = bg_size
        
        # Converting from normalized device coordinates to 0-1 range
        NDC_x = (NDC_x + 1) / 2
        NDC_y = 1 - ((NDC_y + 1) / 2)
        
        # Converting from 0-1 range to pixel coordinates
        self.size = int((x_size / 2) * width)
        self.screen_x = int(NDC_x * width)
        self.screen_y = int(NDC_y * height)
      
        
class SignObject(object):
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
        Applies perspective projection to the sign object, and returns the coordinates of the top-left and bottom-right
        corners of the sign object in the image.
        """
        # Scale the sign coordinates by the perspective projection matrix
        x1, y1, _, w = proj_matrix.dot(np.array([self.x1, self.y1, self.z, 1]))
        x2, y2, _, w = proj_matrix.dot(np.array([self.x2, self.y2, self.z, 1]))
        
        # Transform to normalized device coordinates
        x1f, y1f, x2f, y2f = x1/w, y1/w, x2/w, y2/w
        
        x_size = abs(x2f - x1f)
        return Anchor(bg_size, NDC_x=x1f, NDC_y=y1f, x_size=x_size)


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
    

def produce_anchors(bg_size, size, x, y, sign_near_dist, sign_far_dist):
    anchors = {}
    height, width, _ = bg_size
    aspect_ratio = width / height
    proj_matrix = create_perspective(FOVY, aspect_ratio, NEAR_CLIPPING_PLANE_DIST, FAR_CLIPPING_PLANE_DIST)
    
    # Projection matrix assumes negative z values in front of camera
    sign_near_z = -1 * sign_near_dist
    sign_far_z = -1 * sign_far_dist
    
    sign_near = SignObject(x, y, z=sign_near_z, size=size)
    sign_far = SignObject(x, y, z=sign_far_z, size=size)
    
    anchors['near'] = sign_near.perspective_transform(bg_size, proj_matrix)
    anchors['far'] = sign_far.perspective_transform(bg_size, proj_matrix)
    return anchors
    

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
  
  
# World coordinates for sign objects
sign_coords = {'x':1.5, 'y':1, 'near_dist':4, 'far_dist':20, 'size':0.5}


def get_view_plane_bounds(distance, fovy, aspect_ratio):
    top = distance * math.tan(math.radians(fovy) / 2)
    right = top * aspect_ratio
    left = -right
    bottom = -top
    return {'top':top, 'right':right, 'left':left, 'bottom':bottom}


def draw_anchors(sign_path, bg_path, fg_size, x, y, near_dist, far_dist):
    sign_img = cv2.imread(sign_path)
    bg_img = cv2.imread(bg_path)
    
    anchors = produce_anchors(bg_img.shape, fg_size, x, y, near_dist, far_dist)
    sign_img_near = cv2.resize(sign_img, (anchors['near'].size, anchors['near'].size))
    sign_img_far = cv2.resize(sign_img, (anchors['far'].size, anchors['far'].size))
    final_img = overlay(sign_img_far, bg_img, anchors['far'].screen_x, anchors['far'].screen_y)
    final_img = overlay(sign_img_near, final_img, anchors['near'].screen_x, anchors['near'].screen_y)
    print(f'X: {anchors["near"].screen_x}, Y: {anchors["near"].screen_y}, Size: {anchors["near"].size}')
    cv2.imshow('image', final_img)
    
    
# Add slider callbacks
def z_on_change(val):
    z_prop = val / Z_TRACKBAR_MAX
    sign_coords['near_dist'] = z_prop * FAR_CLIPPING_PLANE_DIST + NEAR_CLIPPING_PLANE_DIST + 0.5
    draw_anchors(sign_path, bg_path, sign_coords['size'], sign_coords['x'], sign_coords['y'], 
                sign_coords['near_dist'], sign_coords['far_dist'])
    
    
def x_on_change(val):
    x_prop = 2 * val / X_TRACKBAR_MAX - 1
    right_bound = get_view_plane_bounds(sign_coords['near_dist'], FOVY, aspect_ratio)['right']
    world_x = x_prop * right_bound
    sign_coords['x'] = min(max(world_x, -right_bound), right_bound - sign_coords['size'])
    draw_anchors(sign_path, bg_path, sign_coords['size'], sign_coords['x'], sign_coords['y'], 
                sign_coords['near_dist'], sign_coords['far_dist'])
    
    
def y_on_change(val):
    y_prop = 2 * val / Y_TRACKBAR_MAX - 1
    upper_bound = get_view_plane_bounds(sign_coords['near_dist'], FOVY, aspect_ratio)['top']
    world_y = y_prop * upper_bound
    sign_coords['y'] = min(max(world_y, -upper_bound + sign_coords['size']), upper_bound)
    draw_anchors(sign_path, bg_path, sign_coords['size'], sign_coords['x'], sign_coords['y'], 
                sign_coords['near_dist'], sign_coords['far_dist'])   

            
current_dir = os.path.dirname(os.path.realpath(__file__))
bg_path = os.path.join(current_dir, 'Backgrounds/GTSDB/00014.png')
sign_path = os.path.join(current_dir, 'Signs/0.jpg')
img = cv2.imread(bg_path)
height, width, channels = img.shape
aspect_ratio = width / height

cv2.imshow('image', img)

Z_TRACKBAR_MAX = 100
X_TRACKBAR_MAX = 200
Y_TRACKBAR_MAX = 200

cv2.createTrackbar('z', 'image', 0, Z_TRACKBAR_MAX, z_on_change)
cv2.createTrackbar('x', 'image', 0, X_TRACKBAR_MAX, x_on_change)
cv2.createTrackbar('y', 'image', 0, Y_TRACKBAR_MAX, y_on_change)

cv2.waitKey(0)