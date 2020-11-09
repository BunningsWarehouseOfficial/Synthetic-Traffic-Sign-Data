"""Module of functions to apply damage to signs."""
# Co-author: Jack Downes
# https://github.com/ai-research-students-at-curtin/sign_augmentation
# Last modified: 9 November 2020

import random as rand
import math
import numpy as np
import cv2 as cv
from skimage import draw
from utils import overlay, calc_ratio, calc_quadrant_diff

attributes = {
    "damage" : "None",
    "tag"    : "-1",  # Set of parameters used to generate damage as string 
    "ratio"  : 0,     # Quantity of damage (0 for no damage, 1 for all damage)
    }


def validate_sign(img):
    """Ensure sign image fits within the parameters of valid signs."""
    has_alpha = (img.shape[2] == 4)
    if has_alpha:
        return img
    else:
        raise ValueError


def no_damage(img):
    """Return the image as is, along with its attributes."""
    dmg = validate_sign(img)
    height, width, _ = img.shape

    # Assign labels
    att = attributes
    att["damage"] = "no_damage"
    att["tag"]    = "-1"
    att["ratio"]  = calc_ratio(dmg, img)  # This should be 0

    return dmg, att


def remove_quadrant(img):
    """Make one random quandrant transparent."""
    dmg = validate_sign(img)
    quadrant = dmg[:,:,3].copy()

    height, width, _ = img.shape
    centre_x = int(round( width / 2 ))
    centre_y = int(round( height / 2 ))

    quad_num = rand.randint(1, 4)  # For selecting which quadrant it will be
    # Remove the quadrant: -1 offset is necessary to avoid damaging part of a wrong quadrant
    if quad_num is 1:         # top-right          centre
        cv.rectangle(quadrant, (width, 0), (centre_x, centre_y-1), (0,0,0), -1)
    elif quad_num is 2:       # top-left           centre
        cv.rectangle(quadrant, (0, 0), (centre_x-1, centre_y-1), (0,0,0), -1)
    elif quad_num is 3:       # bottom-left        centre
        cv.rectangle(quadrant, (0, height), (centre_x-1, centre_y), (0,0,0), -1)
    elif quad_num is 4:       # bottom-right       centre
        cv.rectangle(quadrant, (width, height), (centre_x, centre_y), (0,0,0), -1)
    
    dmg = cv.bitwise_and(img, img, mask=quadrant)
    ratios = calc_quadrant_diff(dmg, img)

    # Assign labels
    att = attributes
    att["damage"] = "quadrant"
    att["tag"]    = str(quad_num)
    att["ratio"]  = "{:.3f}".format(1 - calc_ratio(dmg, img))  # This should be around 0.25

    return dmg, att


def remove_hole(img):
    """Remove a circle-shaped region from the edge of the sign."""
    dmg = validate_sign(img)
    hole = dmg[:,:,3].copy()
    att = attributes

    height, width, _ = dmg.shape
    centre_x = int(round( width / 2 ))
    centre_y = int(round( height / 2 ))

    angle = rand.randint(0, 359)  # Angle from x-axis, counter-clockwise through quadrant I
    radius = int(2 * height / 5)
    rad = -(angle * math.pi / 180)  # Radians
    x = centre_x + int(radius * math.cos(rad))   # x-coordinate of centre
    y = centre_y + int(radius * math.sin(rad))   # y-coordinate of centre

    cv.circle(hole, (x,y), radius, (0,0,0), -1)  # -1 to create a filled circle
    dmg = cv.bitwise_and(dmg, dmg, mask=hole)

    # Assign labels
    att = attributes
    att["damage"] = "big_hole"
    att["tag"]    = "_".join((str(angle), str(radius), str(x), str(y)))
    att["ratio"]  = "{:.3f}".format(1 - calc_ratio(dmg, img))

    return dmg, att


def bullet_holes(img):
    """Create randomised bullet holes of varying sizes."""
    painted = validate_sign(img).copy()
    bullet_holes = painted[:,:,3].copy()
    att = attributes

    height, width, _ = img.shape

    # Apply damage to 'painted' and 'bullet_holes'
    num_holes = rand.randint(7, 30)  # Number of holes on sign
    for x in range(num_holes):
        x = rand.randint(0, height)  # x and y coordinates
        y = rand.randint(0, height)
        min_size, max_size = int(round(width*0.01)), int(round(width*0.03))
        size = rand.randint(min_size, max_size)  # Size of bullet hole
        annulus = rand.uniform(1.6, 2.2)  # Multiplication factor to determine size of annulus
        an = 200  # Colour of damaged 'paint' outer annulus
        hl = rand.randint(0, 150)  # How dark the hole is if it DIDN'T penetrate
        
        # Paint the ring around the bullet hole first
        cv.circle(painted, (x,y), int(size * annulus), (an,an,an,255), -1)
        # If the bullet didn't penetrate through the sign, grey out rather than making transparent
        if (size < 6):
            cv.circle(painted, (x,y), size, (hl,hl,hl,255), -1)
        else:
            cv.circle(bullet_holes, (x,y), size, (0,0,0), -1)
    
    dmg = cv.bitwise_and(painted, painted, mask=bullet_holes)

    # Assign labels
    att = attributes
    att["damage"] = "bullet_holes"
    att["tag"]    = str(num_holes)
    att["ratio"]  = "{:.3f}".format(1 - calc_ratio(dmg, dmg))

    return dmg, att


### The following methods are all for graffiti damage ###

def in_bounds(img, xx, yy):
    """Return true if (xx,yy) is in bounds of the sign."""
    height, width, _ = img.shape
    alpha_ch = cv.split(img)[3]
    in_bounds = False

    # Check that the points are within the image
    if (xx > 0) and (xx < width) and (yy > 0) and (yy < height):
        # Test the opacity to check if it's in bounds of the sign
        if alpha_ch[yy][xx] > 0:
            in_bounds = True
    return in_bounds

def calc_points(img, x0, y0, offset):
    """Caclulate a random midpoint and endpoint for the bezier curve."""
    # Choose the midpoint for the bezier curve
    x1, y1 = 0, 0
    while not in_bounds(img, x1, y1):
        x1 = x0 + rand.randint(-offset, offset)
        y1 = y0 + rand.randint(-offset, offset)

    # Choose the end point
    x2, y2 = 0, 0
    while not in_bounds(img, x2, y2):
        x2 = x1 + rand.randint(-offset, offset)
        y2 = y1 + rand.randint(-offset, offset)
    
    return x1, y1, x2, y2

def draw_bezier(grft, x0, y0, x1, y1, x2, y2, thickness, color):
    """Draw a single bezier curve of desired thickness and colour."""
    alpha = rand.randint(240, 255)  # Random level of transparency
    # skimage.draw.bezier_curve() only draws curves of thickness 1 pixel,
    # We layer multiple curves to produce a curve of desired thickness
    start = -(thickness // 2)
    end = thickness // 2
    for ii in range(start, end+1):
        # Draw curves, shifting vertically and adjusting x-coordinates to round the edges
        yy, xx = draw.bezier_curve(y0+ii, x0+abs(ii), y1+ii, x1, y2+ii, x2-abs(ii), weight=1, shape=grft.shape)
        grft[yy, xx] = color + (alpha,)  # Modify the colour of the pixels that belong to the curve

def graffiti(img, color=(0,0,0), initial=0.1, final=0.4, step=0.1):
    """Apply graffiti damage to sign.
       :param initial: the first target level of obscurity (0-1)
       :param final: the level of obscurity to stop at (0-1)
       :param step: the step for the next level of obscurity
       :returns: a list containing the damaged images, and a list with the corresponding attributes"""
    validate_sign(img)
    height, width, _ = img.shape
    grft = np.zeros((height, width, 4), dtype=np.uint8)  # New blank image for drawing the graffiti on.
    grfts = []  # To hold the graffiti layers
    attrs = []  # To hold their corresponding attributes
    dmgs = []   # To hold the damaged signs

    ratio = 0.0
    x0, y0 = width//2, height//2  # Start drawing in the centre of the image
    # Multiply by 100 to use integers in the for loop
    initial = int(initial * 100)
    final = int(final * 100)
    step = int(step * 100)

    # Loop for the number of target obscurities
    for target in range(initial, final+1, step):
        # Keep drawing bezier curves until the obscurity hits the target
        while ratio < target:
            radius = width // 5  # Radius of max distance to the next point
            x1, y1, x2, y2 = calc_points(img, x0, y0, radius)
            thickness = int(round( width // 20 ))
            draw_bezier(grft, x0, y0, x1, y1, x2, y2, thickness, color)
            # Make the end point the starting point for the next curve
            x0, y0 = x2, y2
            ratio = round( calc_ratio(grft, img) * 100, 1 )
        # Add a copy to the list and continue layering more graffiti
        grfts.append(grft.copy())
        # Assign labels
        att = attributes
        att["damage"] = "graffiti"
        att["tag"]    = str(ratio)
        att["ratio"]  = "{:.3f}".format(ratio / 100)
        attrs.append(att.copy())
    
    k = (int(round( width/20 )) // 2) * 2 + 1  # Kernel size must be odd
    for grft in grfts:
        # Apply a Gaussian blur to each image, to smooth the edges
        grft = cv.GaussianBlur(grft, (k,k), 0)

        # Combine with the original alpha channel to remove anything that spilt over the sign
        grft[:,:,3] = cv.bitwise_and(grft[:,:,3], img[:,:,3])
        dmg = overlay(grft, img)
        dmgs.append(dmg.copy())

    return dmgs, attrs


### The following two functions are both for bend ###

def combine(img1, img2):
    """Combine the left half of img1 with right half of img2 and returns the result."""
    _,wd,_ = img1.shape
    result = img1.copy()
    # Save the alpha data (to replace later), as convertScaleAbs() will affect the transparency
    alpha_ch = cv.split(result)[3]
    
    # Darken the entire image of the copy
    alpha = 1   # No change to contrast
    beta = -20  # Decrease brightness
    cv.convertScaleAbs(result, result, alpha, beta)
    # Replace the alpha data
    result[:,:,3] = alpha_ch
    
    # Copy over the right half of img2 onto the darkened image
    result[:,wd//2:wd] = img2[:,wd//2:wd]

    return result

def bend_vertical(img):
    """Apply perspective warp to tilt images and combine to produce bent signs.
       :returns: a list of bent signs and a list of corresponding attributes"""
    dmg = validate_sign(img)
    ht,wd,_ = img.shape  # Retrieve image dimentions
    dmgs = []
    attrs = []

    # TILT 1
    pt = wd // 24
    xx = pt * 3
    yy = pt // 2
    # Keep top-middle and bottom-middle unchanged to bend on the vertical axis
    # Right             Top-left Top-middle  Bottom-left Bottom-middle
    src = np.float32( [ [pt,pt], [wd//2,pt], [pt,ht-pt], [wd//2,ht-pt] ] )
    dst = np.float32( [ [xx,yy], [wd//2,pt], [xx,ht-yy], [wd//2,ht-pt] ] )
    matrix = cv.getPerspectiveTransform(src, dst)
    right = cv.warpPerspective(img, matrix, (wd,ht))
    # Left              Top-middle  Top-right   Bottom-middle  Bottom-right
    src = np.float32( [ [wd//2,pt], [wd-pt,pt], [wd//2,ht-pt], [wd-pt,ht-pt] ] )
    dst = np.float32( [ [wd//2,pt], [wd-xx,yy], [wd//2,ht-pt], [wd-xx,ht-yy] ] )
    matrix = cv.getPerspectiveTransform(src, dst)
    left = cv.warpPerspective(img, matrix, (wd,ht))
    
    # Combine the right tilt with the original forward-facing image
    dmg = combine(img, right)
    dmgs.append(dmg.copy())
    att = attributes
    att["damage"] = "bend"
    att["tag"]    = "0_40"
    att["ratio"]  = "{:.3f}".format(1 - calc_ratio(dmg, img))
    attrs.append(att.copy())

    # Combine the left tilt with the original forward-facing image
    dmg = combine(left, img)
    dmgs.append(dmg.copy())
    att["damage"] = "bend"
    att["tag"]    = "40_0"
    att["ratio"]  = "{:.3f}".format(1 - calc_ratio(dmg, img))
    attrs.append(att.copy())

    # Combine the left and right tilt
    dmg = combine(left, right)
    dmgs.append(dmg.copy())
    att["damage"] = "bend"
    att["tag"]    = "40_40"
    att["ratio"]  = "{:.3f}".format(1 - calc_ratio(dmg, img))
    attrs.append(att.copy())

    # TILT 2
    # pt = wd // 12
    # xx = pt * 3
    # yy = pt // 2
    # # Right             Top-left Top-middle  Bottom-left Bottom-middle
    # src = np.float32( [ [pt,pt], [wd//2,pt], [pt,ht-pt], [wd//2,ht-pt] ] )
    # dst = np.float32( [ [xx,yy], [wd//2,pt], [xx,ht-yy], [wd//2,ht-pt] ] )
    # matrix = cv.getPerspectiveTransform(src, dst)
    # right = cv.warpPerspective(img, matrix, (wd,ht))
    # # Left              Top-middle  Top-right   Bottom-middle  Bottom-right
    # src = np.float32( [ [wd//2,pt], [wd-pt,pt], [wd//2,ht-pt], [wd-pt,ht-pt] ] )
    # dst = np.float32( [ [wd//2,pt], [wd-xx,yy], [wd//2,ht-pt], [wd-xx,ht-yy] ] )
    # matrix = cv.getPerspectiveTransform(src, dst)
    # left = cv.warpPerspective(img, matrix, (wd,ht))
    
    # # Combine the left and right tilt with the original forward-facing image
    # dmg["0_60"] = combine(img, right)
    # dmg["60_0"]  = combine(left, img)
    # # Combine the left and right tilt
    # dmg["60_60"] = combine(left, right)

    return dmgs, attrs