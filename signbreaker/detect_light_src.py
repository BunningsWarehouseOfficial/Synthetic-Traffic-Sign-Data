import os
import cv2
import numpy as np
import time
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)


if __name__ == '__main__':
    img = cv2.imread('Backgrounds/GTSDB/00029.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    height, width = luminance.shape
    luminance = np.reshape(luminance, (-1)) # Flatten
    indices = np.arange(len(luminance))
    
    SAMPLING_CONSTANT = 0.1    # Percentage of image pixels that are sampled
    NUM_SEGMENTS = 10          # How many vertical segments to divide the image into
    
    segment_width = int(width / NUM_SEGMENTS)
    print(segment_width)
    
    num_iterations = 200
    sample_size = int(img.shape[0] * img.shape[1] * SAMPLING_CONSTANT)
    maxima_counter = Counter()
    
    # Use Montecarlo sampling to find maxima
    t1 = time.time()
    for _ in range(num_iterations):
        samples = np.random.choice(indices, sample_size)
        max_ind = max(samples, key=luminance.__getitem__)
        col = max_ind % width
        col = round(col / segment_width) * segment_width
        maxima_counter[col] += 1
    print(maxima_counter.most_common(10)) 
    print('Time taken for sampling: {}'.format(time.time() - t1))
    
    for col, count in maxima_counter.most_common(10):
        cv2.circle(img, (col, int(height * 0.5)), int(0.5 * count), color=(0, 255, 0), thickness=-1)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    
    
    