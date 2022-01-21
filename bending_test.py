import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from signbreaker.damage import bend_vertical

current_dir = os.path.dirname(os.path.realpath(__file__))
    

if __name__ == "__main__":
    img = cv.imread(os.path.join(current_dir, 'signbreaker/Sign_Templates/2_Processed/1.png'), cv.IMREAD_UNCHANGED)
    dims = img.shape[:2]
    
    imgs, attrs = bend_vertical(img)
    cv.imwrite(filename='bend_vertical_right.png', img=imgs[0])
    cv.imwrite(filename='bend_vertical_left.png', img=imgs[1])
    cv.imwrite(filename='bend_vertical_both.png', img=imgs[2])
    
        
    
    