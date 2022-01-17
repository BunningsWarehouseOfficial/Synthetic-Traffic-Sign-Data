import cv2
import math
from dataset_gen.sequence_gen.create_sequences_auto import SIGN_COORDS, produce_anchors
from utils import overlay


class ShowAnchors(object):
    """[summary]
    A class which is a wrapper for the functions needed to show num_frames projected
    signs at varying distances from the camera, using cv2.imshow
    """
    def __init__(self, bg_path, fg_path, min_dist, max_dist, num_frames, fovy):
        self.bg_img = cv2.imread(bg_path)
        self.fg_img = cv2.imread(fg_path)
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_frames = num_frames
        self.fovy = fovy
        
        self.display_img = self.bg_img.copy()
        height, width, _ = self.bg_img.shape
        self.aspect_ratio = width / height
        
        self.x_trackbar_max = 200
        self.y_trackbar_max = 200
        
        self.__draw_anchors(SIGN_COORDS['size'], SIGN_COORDS['x'], SIGN_COORDS['y'], self.min_dist, self.max_dist,
                          self.num_frames)
        cv2.imwrite('overlayed_sequence_auto.jpg', self.display_img)
        cv2.createTrackbar('x', 'image', 0, self.x_trackbar_max, self.__x_on_change)
        cv2.createTrackbar('y', 'image', 0, self.y_trackbar_max, self.__y_on_change)
        cv2.waitKey(0)
        

    def __get_view_plane_bounds(self, distance, fovy, aspect_ratio):
        top = distance * math.tan(math.radians(fovy) / 2)
        right = top * aspect_ratio
        left = -right
        bottom = -top
        return {'top':top, 'right':right, 'left':left, 'bottom':bottom}


    def __draw_anchors(self, fg_size, x, y, min_dist, max_dist, num_frames):
        self.display_img = self.bg_img.copy()
        anchors = produce_anchors(self.bg_img.shape, fg_size, x, y, min_dist, max_dist, num_frames)
        
        for anchor in anchors:
            sign_img_scaled = cv2.resize(self.fg_img, (anchor.size, anchor.size))
            self.display_img = overlay(sign_img_scaled, self.display_img, anchor.screen_x, anchor.screen_y)
        cv2.imshow('image', self.display_img)
        
        
    def __x_on_change(self, val):
        x_prop = 2 * val / self.x_trackbar_max - 1
        right_bound = self.__get_view_plane_bounds(self.min_dist, self.fovy, self.aspect_ratio)['right']
        SIGN_COORDS['x'] = x_prop * right_bound
        self.__draw_anchors(SIGN_COORDS['size'], SIGN_COORDS['x'], SIGN_COORDS['y'], 
                        self.min_dist, self.max_dist, self.num_frames)
        
        
    def __y_on_change(self, val):
        y_prop = 2 * val / self.y_trackbar_max - 1
        upper_bound = self.__get_view_plane_bounds(self.min_dist, self.fovy, self.aspect_ratio)['top']
        SIGN_COORDS['y'] = y_prop * upper_bound
        self.__draw_anchors(SIGN_COORDS['size'], SIGN_COORDS['x'], SIGN_COORDS['y'], 
                        self.min_dist, self.max_dist, self.num_frames)   