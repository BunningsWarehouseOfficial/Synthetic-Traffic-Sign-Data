import os
import cv2
import numpy as np
import random
import math
from collections import Counter

from damage import bend_vertical
from utils import overlay

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

SAMPLING_CONSTANT = 0.1    # Percentage of image pixels that are sampled
NUM_SEGMENTS = 10          # Number of segments horizontally
NUM_ITERATIONS = 200       # How many iterations to run the montecarlo sampling


def find_light_source(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    _, width = luminance.shape
    luminance = np.reshape(luminance, (-1)) # Flatten
    indices = np.arange(len(luminance))
    
    segment_width = int(width / NUM_SEGMENTS)
    sample_size = int(img.shape[0] * img.shape[1] * SAMPLING_CONSTANT)
    maxima_counter = Counter()

    # Use Montecarlo sampling to find maxima
    for _ in range(NUM_ITERATIONS):
        samples = np.random.choice(indices, sample_size)
        max_ind = max(samples, key=luminance.__getitem__)
        col = round(max_ind % width / segment_width) * segment_width
        row = round(max_ind // width / segment_width) * segment_width
        maxima_counter[(col, row)] += 1

    # for (x, y), count in maxima_counter.most_common(10):
    #     cv2.circle(img, (x, y), int(0.5 * count), color=(0, 255, 0), thickness=-1)
    
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imshow('win', img)
    # cv2.waitKey(0)
    return maxima_counter.most_common(1)[0]


if __name__ == '__main__':
    bg_img = cv2.imread('Backgrounds/GTSDB/00029.png', cv2.IMREAD_UNCHANGED)[10:-10, 10:-10]
    fg_img = cv2.imread('Sign_Templates/2_Processed/1.png', cv2.IMREAD_UNCHANGED)
    (light_x, light_y), intensity = find_light_source(bg_img.copy())
    
    bg_height, bg_width = bg_img.shape[:2]
    fg_height, fg_width = fg_img.shape[:2]
    
    current_ratio = fg_width / bg_width  
    target_ratio = random.uniform(0.033, 0.066)  
    scale_factor = target_ratio / current_ratio
    new_size = int(fg_width * scale_factor)
    
    # Randomise sign placement within middle third of background
    fg_x = random.randint(0, bg_width - new_size)
    third = bg_height // 3
    fg_y = random.randint(third, bg_height - third)
    
    axis_angle = 0
    bend_angle = 30
    
    # Calculate beta difference
    vec1 = np.array([math.cos(math.radians(90-axis_angle)), math.sin(math.radians(90-axis_angle))])
    vec2 = np.array([fg_x - light_x, fg_y - light_y])
    angle = math.acos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    beta_diff = math.copysign(1, vec2[0]) * math.sin(angle) * intensity * (bend_angle / 90) * 4
    
    dmg, att = bend_vertical(fg_img, axis_angle, bend_angle, beta_diff=beta_diff)
    
    fg_img = cv2.resize(dmg, (new_size, new_size), interpolation=cv2.INTER_AREA)
    
    image = overlay(fg_img, bg_img, fg_x, fg_y)
    cv2.imshow('win', image)
    cv2.waitKey(0)
    
    
    