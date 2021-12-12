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
import os
import math
from utils import overlay

NEAR_CLIPPING_PLANE_DIST = 2
FAR_CLIPPING_PLANE_DIST = 50
FOVY = 60


class Anchor(object):
    def __init__(self, bg_size, NDC_x, NDC_y, x_size, y_size):
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
        
    def __str__(self):
        return f"Anchor: {self.screen_x}, {self.screen_y}, {self.size}"
      
        
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
        y_size = abs(y2f - y1f)
        return Anchor(bg_size, NDC_x=x1f, NDC_y=y1f, x_size=x_size, y_size=y_size)


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
    

def produce_anchor(bg_size, size, x, y, sign_dist):
    anchors = {}
    height, width, _ = bg_size
    aspect_ratio = width / height
    proj_matrix = create_perspective(FOVY, aspect_ratio, NEAR_CLIPPING_PLANE_DIST, FAR_CLIPPING_PLANE_DIST)
    
    # Projection matrix assumes negative z values in front of camera
    sign_z = -1 * sign_dist
    
    sign_near = SignObject(x, y, z=sign_z, size=size)
    
    return sign_near.perspective_transform(bg_size, proj_matrix)
  
  
# World coordinates for sign objects
sign_coords = {'x':1.5, 'y':1, 'dist':4, 'size':0.5}


def get_view_plane_bounds(distance, fovy, aspect_ratio):
    top = distance * math.tan(math.radians(fovy) / 2)
    right = top * aspect_ratio
    left = -right
    bottom = -top
    return {'top':top, 'right':right, 'left':left, 'bottom':bottom}


def draw_anchors(sign_path, bg_path, fg_size, x, y, dist):
    sign_img = cv2.imread(sign_path)
    bg_img = cv2.imread(bg_path)
    
    anchor = produce_anchor(bg_img.shape, fg_size, x, y, dist)
    sign_img_near = cv2.resize(sign_img, (anchor.size, anchor.size))
    final_img = overlay(sign_img_near, bg_img, anchor.screen_x, anchor.screen_y)
    print(f'X: {anchor.screen_x}, Y: {anchor.screen_y}, Size: {anchor.size}')
    cv2.imshow('image', final_img)
    
    
# Add slider callbacks
def z_on_change(val):
    z_prop = val / Z_TRACKBAR_MAX
    sign_coords['dist'] = z_prop * FAR_CLIPPING_PLANE_DIST + NEAR_CLIPPING_PLANE_DIST + 0.5
    draw_anchors(sign_path, bg_path, sign_coords['size'], sign_coords['x'], sign_coords['y'], 
                sign_coords['dist'])
    
    
def x_on_change(val):
    x_prop = 2 * val / X_TRACKBAR_MAX - 1
    right_bound = get_view_plane_bounds(sign_coords['dist'], FOVY, aspect_ratio)['right']
    sign_coords['x'] = x_prop * right_bound
    draw_anchors(sign_path, bg_path, sign_coords['size'], sign_coords['x'], sign_coords['y'], 
                sign_coords['dist'])
    
    
def y_on_change(val):
    y_prop = 2 * val / Y_TRACKBAR_MAX - 1
    upper_bound = get_view_plane_bounds(sign_coords['dist'], FOVY, aspect_ratio)['top']
    sign_coords['y'] = y_prop * upper_bound
    draw_anchors(sign_path, bg_path, sign_coords['size'], sign_coords['x'], sign_coords['y'], 
                sign_coords['dist'])   

            
current_dir = os.path.dirname(os.path.realpath(__file__))
bg_path = os.path.join(current_dir, 'Backgrounds/GTSDB/00027.png')
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