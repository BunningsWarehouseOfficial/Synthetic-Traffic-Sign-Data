"""Module of functions to apply damage to signs."""
# Authors: Kristian Rados, Seana Dale, Jack Downes
# https://github.com/ai-research-students-at-curtin/sign_augmentation

import os
import imutils
import random as rand
import math
import numpy as np
import cv2 as cv
from skimage import draw
from utils import calc_damage_ssim, overlay, calc_damage, calc_damage_ssim, calc_damage_sectors, calc_ratio, remove_padding
import ntpath
from synth_image import SynthImage

attributes = {
    "damage_type" : "None",
    "tag"    : "-1",  # Set of parameters used to generate damage as string 
    "damage_ratio"  : 0.0,     # Quantity of damage (0 for no damage, 1 for all damage)
    }

params = {}

def damage_image(image_path, output_dir, config):
    """Applies all the different types of damage to the imported image, saving each one"""
    damaged_images = []
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    img = img.astype('uint8')
    params.update(config)
    
    # Create file writing info: filename, class number, output directory, and labels directory
    _, filename = ntpath.split(image_path)  # Remove parent directories to retrieve the image filename
    class_num, _ = filename.rsplit('.', 1)  # Remove extension to get the sign/class number

    output_path = os.path.join(output_dir, class_num)
    # Create the output directory if it doesn't exist already
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)

    types = config['damage_types']

    def simple_damage(dmg, att):
        """Helper function to avoid repetition."""
        dmg_path = os.path.join(output_path, class_num + "_" + att["damage_type"] + ".png")
        cv.imwrite(dmg_path, dmg)
        damaged_images.append(SynthImage(
            dmg_path, int(class_num), att["damage_type"], att["tag"], float(att["damage_ratio"]), att["sector_damage"]))

    # ORIGINAL UNDAMAGED
    if 'original' in types:
        dmg, att = no_damage(img)
        simple_damage(dmg, att)

    # QUADRANT
    if 'quadrant' in types:
        dmg, att = remove_quadrant(img)
        simple_damage(dmg, att)
    
    # BIG HOLE
    if 'big_hole' in types:
        dmg, att = remove_hole(img)
        simple_damage(dmg, att)
    
    # RANDOMISED BULLET HOLES
    b_conf = config['bullet_holes']
    if 'bullet_holes' in types:
        dmg, att = bullet_holes(img, b_conf['min_holes'], b_conf['max_holes'], b_conf['target'])
        simple_damage(dmg, att)

    #TODO: Only feed unshaded (and bent) half of sign to damage calc for single bent? Idk about double bent
    # BEND
    bd_config = config['bend']
    if 'bend' in types:
        dmgs, attrs = bend_vertical(img, axis_angle=bd_config['axis_angle'], bend_angle=bd_config['bend_angle'], beta_diff=config['beta_diff'])
        for ii in range(len(dmgs)):
            dmg_path = os.path.join(output_path, class_num + "_" + attrs[ii]["damage_type"] + "_" + str(attrs[ii]["tag"]) + ".png")
            cv.imwrite(dmg_path, dmgs[ii])
            damaged_images.append(SynthImage(
                dmg_path, int(class_num), attrs[ii]["damage_type"], attrs[ii]["tag"], float(attrs[ii]["damage_ratio"], attrs[ii]["sector_damage"])))

    # GRAFFITI
    g_conf = config['graffiti']
    if 'graffiti' in types:
        dmgs, attrs = graffiti(img, color=(0,0,0),
                               initial=g_conf['initial'], final=g_conf['final'], step=g_conf['step'])
        for ii in range(len(dmgs)):
            dmg_path = os.path.join(output_path, class_num + "_" + attrs[ii]["damage_type"] + "_" + str(attrs[ii]["damage_ratio"]) + ".png")
            cv.imwrite(dmg_path, dmgs[ii])
            damaged_images.append(SynthImage(
                dmg_path, int(class_num), attrs[ii]["damage_type"], attrs[ii]["tag"], float(attrs[ii]["damage_ratio"], attrs[ii]["sector_damage"])))

    # TINTED YELLOW
    # yellow = np.zeros((height,width,ch), dtype=np.uint8)
    # yellow[:,:] = (0,210,210,255)
    # dmg4 = cv.bitwise_and(img, yellow)
    # # TODO: Use quadrant pixel difference ratio or some other damage metric for labelling?
    # cv.imwrite(os.path.join(output_path, class_num + damage_types[4] + ".png"), dmg4)
    
    # GREY
    # dmg7 = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)   # Convert to greyscale
    # # Threshold the image to get a uniform saturation
    # _, dmg7 = cv.threshold(dmg7, 100, 255, cv.THRESH_BINARY)
    # cv.convertScaleAbs(dmg7, dmg7, alpha=1, beta=200)  # No change to contrast, scale brightness
    # dmg7 = cv.cvtColor(dmg7, cv.COLOR_GRAY2BGRA)   # Convert back to BGRA to add back the alpha channel
    # dmg7[:,:,3] = alpha_ch
    # cv.imwrite(os.path.join(output_path, class_num + damage_types[7] + ".png"), dmg7)
    # TODO: Test with exposure_manipulation()
    # TODO: Write values to file
    
    # TODO: CRACKS (thin crack lines across the sign?)
    # TODO: MISSING SECTIONS (missing polygon sections on edges of sign?)

    return damaged_images


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
    att["damage_type"] = "no_damage"
    att["tag"]    = "-1"
    att["damage_ratio"]  = str(calc_damage(dmg, img))  # This should be 0
    att["sector_damages"] = calc_damage_sectors(dmg, img)

    return dmg, att

def remove_quadrant(img): #TODO: Have quadrant be input parameter, with -1 being random
    """Make one random quandrant transparent."""
    dmg = validate_sign(img)
    quadrant = dmg[:,:,3].copy()

    height, width, _ = img.shape
    centre_x = int(round(width / 2))
    centre_y = int(round(height / 2))

    quad_num = rand.randint(1, 4)  # For selecting which quadrant it will be
    # Remove the quadrant: -1 offset is necessary to avoid damaging part of a wrong quadrant
    if quad_num == 1:         # top-right          centre
        cv.rectangle(quadrant, (width, 0), (centre_x, centre_y-1), (0,0,0), -1)
    elif quad_num == 2:       # top-left           centre
        cv.rectangle(quadrant, (0, 0), (centre_x-1, centre_y-1), (0,0,0), -1)
    elif quad_num == 3:       # bottom-left        centre
        cv.rectangle(quadrant, (0, height), (centre_x-1, centre_y), (0,0,0), -1)
    elif quad_num == 4:       # bottom-right       centre
        cv.rectangle(quadrant, (width, height), (centre_x, centre_y), (0,0,0), -1)
    
    dmg = cv.bitwise_and(img, img, mask=quadrant)

    # Assign labels
    att = attributes
    att["damage_type"] = "quadrant"
    att["tag"]    = str(quad_num)
    att["damage_ratio"]  = "{:.3f}".format(calc_damage(dmg, img))  # This should be around 0.25
    att["sector_damage"] = calc_damage_sectors(dmg, img, num_damage_sectors=params["num_damage_sectors"])

    return dmg, att

def remove_hole(img):
    """Remove a circle-shaped region from the edge of the sign."""
    dmg = validate_sign(img)
    hole = dmg[:,:,3].copy()
    att = attributes

    height, width, _ = dmg.shape
    centre_x = int(round(width / 2))
    centre_y = int(round(height / 2))

    angle = rand.randint(0, 359)  # Angle from x-axis, counter-clockwise through quadrant I
    radius = int(2 * height / 5)
    rad = -(angle * math.pi / 180)  # Radians
    x = centre_x + int(radius * math.cos(rad))   # x-coordinate of centre
    y = centre_y + int(radius * math.sin(rad))   # y-coordinate of centre

    cv.circle(hole, (x,y), radius, (0,0,0), -1)  # -1 to create a filled circle
    dmg = cv.bitwise_and(dmg, dmg, mask=hole)

    # Assign labels
    att = attributes
    att["damage_type"] = "big_hole"
    att["tag"]    = "_".join((str(angle), str(radius), str(x), str(y)))
    att["damage_ratio"]  = "{:.3f}".format(calc_damage(dmg, img))
    att["sector_damage"] = calc_damage_sectors(dmg, img, num_damage_sectors=params["num_damage_sectors"])

    return dmg, att

def bullet_holes(img, min_holes=7, max_holes=30, target=-1):
    """Create randomised bullet holes of varying sizes."""
    painted = validate_sign(img).copy()
    bullet_holes = painted[:,:,3].copy()
    att = attributes

    height, width, _ = img.shape

    min_size, max_size = int(round(width*0.0125)), int(round(width*0.03))
    an = 200  # Colour of damaged 'paint' outer annulus

    # Apply damage until the target damage threshold is reached
    ratio = 0
    if target > 0.0 and target < 1.0:
        # Loop for the number of target obscurities
        num_holes = 0
        while ratio < target:
            x = rand.randint(0, height - 1)  # x and y coordinates
            y = rand.randint(0, height - 1)
            size = rand.randint(min_size, max_size)  # Size of bullet hole
            annulus = rand.uniform(1.6, 2.2)  # Multiplication factor to determine size of annulus
            hl = rand.randint(0, 150)  # How dark the hole is if it DIDN'T penetrate
            
            # Paint the ring around the bullet hole first
            cv.circle(painted, (x,y), int(size * annulus), (an,an,an,255), -1)
            # If the bullet didn't penetrate through the sign, grey out rather than making transparent
            if (size < 6):  #TODO: Make relative to image width
                cv.circle(painted, (x,y), size, (hl,hl,hl,255), -1)
            else:
                cv.circle(bullet_holes, (x,y), size, (0,0,0), -1)
                dmg = cv.bitwise_and(painted, painted, mask=bullet_holes)
            
            ratio = round(calc_damage(painted, img), 3)
            num_holes += 1
    # Apply damage with a random number of holes within the specified min-max range
    else:
        num_holes = rand.randint(min_holes, max_holes)  # Number of holes on sign
        for x in range(num_holes):
            # If the 'hole' (i.e. the imaginary bullet) misses the sign, get new coordinates
            alpha = 0
            while alpha != 255:
                x = rand.randint(0, width - 1)
                y = rand.randint(0, height - 1)
                alpha = bullet_holes[y, x]
            
            size = rand.randint(min_size, max_size)  # Size of bullet hole
            annulus = rand.uniform(1.6, 2.2)  # Multiplication factor to determine size of annulus
            hl = rand.randint(0, 150)  # How dark the hole is if it DIDN'T penetrate
            
            # Paint the ring around the bullet hole first
            cv.circle(painted, (x,y), int(size * annulus), (an,an,an,255), -1)
            # If the bullet didn't penetrate through the sign, grey out rather than making transparent
            if (size < 6):  #TODO: Make relative to image width
                cv.circle(painted, (x,y), size, (hl,hl,hl,255), -1)
            else:
                cv.circle(bullet_holes, (x,y), size, (0,0,0), -1)

        dmg = cv.bitwise_and(painted, painted, mask=bullet_holes)
        ratio = round(calc_damage(painted, img), 3)
    
    # dmg = cv.bitwise_and(painted, painted, mask=bullet_holes)

    # Assign labels
    att = attributes
    att["damage_type"] = "bullet_holes"
    att["tag"]    = str(num_holes)
    att["damage_ratio"]  = "{:.3f}".format(ratio)
    att["sector_damage"] = calc_damage_sectors(painted, img, num_damage_sectors=params["num_damage_sectors"])

    return dmg, att


### The following methods are all for the 'graffiti' damage type ###

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
    grfts = []   # To hold the graffiti layers
    attrs = []   # To hold their corresponding attributes
    targets = [] # To hold their corresponding targets
    dmgs = []    # To hold the damaged signs

    ratio = 0.0
    x0, y0 = width//2, height//2  # Start drawing in the centre of the image
    # Multiply by 100 to use integers in the for loop
    initial = round(initial, 4)
    final = round(final, 4)
    step = round(step, 4)

    # Loop for the number of target obscurities
    target = initial
    while target <= final:
        # Keep drawing bezier curves until the obscurity hits the target
        while ratio < target:
            radius = width // 5  # Radius of max distance to the next point
            x1, y1, x2, y2 = calc_points(img, x0, y0, radius)
            thickness = int(round(width // 20))
            draw_bezier(grft, x0, y0, x1, y1, x2, y2, thickness, color)
            # Make the end point the starting point for the next curve
            x0, y0 = x2, y2
            ratio = round(calc_ratio(grft, img), 4)
        # Add a copy to the list and continue layering more graffiti
        grfts.append(grft.copy())
        targets.append(target)

        target += step
    
    k = (int(round( width/20 )) // 2) * 2 + 1  # Kernel size must be odd
    ii = 0
    for grft in grfts:
        # Apply a Gaussian blur to each image, to smooth the edges
        grft = cv.GaussianBlur(grft, (k,k), 0)
        # Combine with the original alpha channel to remove anything that spilt over the sign
        grft[:,:,3] = cv.bitwise_and(grft[:,:,3], img[:,:,3])
        dmg = overlay(grft, img)
        dmgs.append(dmg.copy())
        
        # Assign labels
        att = attributes
        att["damage_type"] = "graffiti"
        att["tag"]    = str(round(targets[ii], 3))
        att["damage_ratio"]  = "{:.3f}".format(calc_damage(dmg, img))
        # att["damage_ratio"]  = "{:.3f}".format(ratio)
        att["sector_damage"] = calc_damage_sectors(dmg, img, num_damage_sectors=params["num_damage_sectors"])
        attrs.append(att.copy())

        ii += 1
    return dmgs, attrs


### The following three functions are both for the 'bend' damage type ###

def combine(img1, img2, beta_diff=-20):
    """Combine the left half of img1 with right half of img2 and returns the result."""
    _,wd,_ = img1.shape
    result = img1.copy()
    # Save the alpha data (to replace later), as convertScaleAbs() will affect the transparency
    alpha_ch = cv.split(result)[3]
    
    # Darken the entire image of the copy
    alpha = 1  # No change to contrast
    beta = beta_diff  # Decrease brightness
    cv.convertScaleAbs(result, result, alpha, beta)
    # Replace the alpha data
    result[:,:,3] = alpha_ch
    
    # Copy over the right half of img2 onto the darkened image
    result[:,wd//2:wd] = img2[:,wd//2:wd]

    return result

def tilt(img, angle, dims):                               
        #                                           (dst)
        #        0,0 ------> x                      *
        #                                           |\
        #      0,0    (src)                         | \  
        #      |            (wd/2, 0)   (delta_y)-->|  \<--(d)        
        #      |      *_______↓                     |﹍﹍\<---(angle)
        #      |      |       |                     |↑___|_(delta_x)    
        #             |       |     =========>      |    | 
        #      y      |       |                     |    |
        #             |       |                     |    |
        #             |_______|                     |   /
        #             *                             |  /
        #             ↑                             | /
        #            (0, ht-1)                      |/
        #                                           * 
        #                                           
        ht, wd = dims
        angle = max(min(angle, 80), 0)  # Limit the angle to between 0 and 80 degrees
        angle = math.radians(angle)  # Convert to radians
        d = wd // 2 * math.cos(angle)  
        delta_x = int(d * math.cos(angle))
        delta_y = int(d * math.sin(angle))
        xx = wd // 2 - delta_x
        yy = -delta_y
        
        # Keep top-middle and bottom-middle unchanged to bend on the vertical axis
        # Right             Top-left  Top-middle  Bottom-left  Bottom-middle
        src = np.float32( [ [0,0],    [wd//2,0],  [0,ht],      [wd//2,ht] ] )
        dst = np.float32( [ [xx,yy],  [wd//2,0],  [xx,ht-yy],  [wd//2,ht] ] )
        matrix = cv.getPerspectiveTransform(src, dst)
        right = cv.warpPerspective(img, matrix, (wd,ht))

        # Left              Top-middle  Top-right   Bottom-middle  Bottom-right
        src = np.float32( [ [wd//2,0],  [wd,0],     [wd//2,ht],    [wd-0,ht] ] )
        dst = np.float32( [ [wd//2,0],  [wd-xx,yy], [wd//2,ht],    [wd-xx,ht-yy] ] )
        matrix = cv.getPerspectiveTransform(src, dst)
        left = cv.warpPerspective(img, matrix, (wd,ht))
        return left, right

def bend_vertical(img, axis_angle, bend_angle, beta_diff=0):
    """Apply perspective warp to tilt images and combine to produce bent signs.
       :param img: the image to use to produce damaged signs
       :param axis_angle: the bearing of the axis of rotation in x-y plane
       :returns: a list of bent signs and a list of corresponding attributes"""
    bend_img = imutils.rotate_bound(img, axis_angle)
    bend_img = remove_padding(bend_img)
    ht, wd,_ = bend_img.shape  # Retrieve image dimentions
    dmgs = []
    attrs = []
    left, right = tilt(bend_img, bend_angle, (ht, wd))

    def apply_bend(left, right, tag):
        dmg = combine(left, right, beta_diff)
        dmg = imutils.rotate_bound(dmg, -axis_angle)
        dmg = remove_padding(dmg)
        dmgs.append(dmg.copy())
        att = attributes
        att["damage_type"] = "bend"
        att["tag"]    = "{}".format(tag)
        original = cv.resize(remove_padding(img), dmg.shape[:2][::-1])
        att["damage_ratio"]  = "{:.3f}".format(calc_damage_ssim(dmg, original))
        att["sector_damage"] = calc_damage_sectors(dmg, img, num_damage_sectors=params["num_damage_sectors"], method='ssim')
        attrs.append(att.copy())
        
    # Combine the right tilt with the original forward-facing image
    #TODO: Would look more realistic with non-zero beta_diff, but introduces too much complexity with exposure atm
    apply_bend(bend_img, right, 'right')
    # Combine the left tilt with the original forward-facing image
    apply_bend(left, bend_img, 'left')
    # Combine the left and right tilt
    apply_bend(left, right, 'both')
    return dmgs, attrs
