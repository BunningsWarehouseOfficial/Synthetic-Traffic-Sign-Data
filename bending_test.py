import os
import numpy as np
import cv2 as cv

from signbreaker.damage import bend_vertical

current_dir = os.path.dirname(os.path.realpath(__file__))
    

if __name__ == "__main__":
    img = cv.imread(os.path.join(current_dir, 'signbreaker/Sign_Templates/2_Processed/9.png'), cv.IMREAD_UNCHANGED)
    dims = img.shape[:2]
    
    imgs, attrs = bend_vertical(img, 40)
    cv.imwrite(filename='bend_vertical_right_pruned.png', img=imgs[0])
    cv.imwrite(filename='bend_vertical_left_pruned.png', img=imgs[1])
    cv.imwrite(filename='bend_vertical_both_pruned.png', img=imgs[2])
    cv.imwrite(filename='original.png', img=img)
    print(attrs)
    
        
    
    