"""A wrapper object to find the location coordinates and intensity of the most predominant light source in an image via
Monte Carlo sampling, storing them as instance variables."""

import cv2
import numpy as np
from collections import Counter

SAMPLING_CONSTANT = 0.1  # Percentage of image pixels that are sampled
NUM_SEGMENTS = 10        # Number of segments horizontally
NUM_ITERATIONS = 200     # How many iterations to run the montecarlo sampling


class BgImage:
    def __init__(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.path = image_path
        self.shape = image.shape[:2]
        self.find_light_source(image)
        

    # TODO: At some point make sure this works for images other than the GTSDB ones we have been using for development
    def find_light_source(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[10:-10, 10:-10]
        luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        _, width = luminance.shape
        luminance = np.reshape(luminance, (-1)) # Flatten
        indices = np.arange(len(luminance))
        
        segment_width = int(width / NUM_SEGMENTS)
        sample_size = int(img.shape[0] * img.shape[1] * SAMPLING_CONSTANT)
        maxima_counter = Counter()

        # Use Monte Carlo sampling to find maxima
        for _ in range(NUM_ITERATIONS):
            samples = np.random.choice(indices, sample_size)
            max_ind = max(samples, key=luminance.__getitem__)
            col = round(max_ind % width / segment_width) * segment_width
            row = round(max_ind // width / segment_width) * segment_width
            maxima_counter[(col, row)] += 1
            
        (light_x, light_y), intensity = maxima_counter.most_common(1)[0]
        self.light_coords = (light_x, light_y)
        self.light_intensity = intensity
        