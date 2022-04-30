"""Module of functions to apply damage to signs."""
# Extended from original functions by Jack Downes: https://github.com/ai-research-students-at-curtin/sign_augmentation

import os
import imutils
import random as rand
import math
import ntpath
import numpy as np
import cv2 as cv
from pathlib import Path
from skimage import draw
from utils import calc_damage_ssim, overlay, calc_damage, calc_damage_ssim, calc_damage_sectors, calc_ratio, remove_padding, pad
from synth_image import SynthImage

attributes = {
    "damage_type" : "None",
    "tag"    : "-1",  # Set of parameters used to generate damage as string 
    "damage_ratio"  : 0.0, # Quantity of damage (0 for no damage, 1 for all damage)
    }

dmg_measure = "pixel_wise"
num_sectors = 4


def damage_image(image_path, output_dir, config, backgrounds=[]):
    """Applies all the different types of damage to the imported image, saving each one"""
    damaged_images = []
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    img = img.astype('uint8')
    
    # Create file writing info: filename, class number, output directory, and labels directory
    _, filename = ntpath.split(image_path)  # Remove parent directories to retrieve the image filename
    class_num, _ = filename.rsplit('.', 1)  # Remove extension to get the sign/class number

    output_path = os.path.join(output_dir, class_num)
    # Create the output directory if it doesn't exist already
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)

    num_damages = config['num_damages']
    global dmg_measure; dmg_measure = config['damage_measure_method']
    global num_sectors; num_sectors = config['num_damage_sectors']

    def apply_damage(dmg, att):
        """Helper function to avoid repetition."""
        dmg_path = os.path.join(output_path, f"{class_num}_{att['damage_type']}")
        if att['tag']:
            dmg_path += f"_{att['tag']}.png"
        else:
            dmg_path += ".png"
        cv.imwrite(dmg_path, dmg)
        damaged_images.append(SynthImage(
            dmg_path, int(class_num), att["damage_type"], att["tag"], float(att["damage_ratio"]), att["sector_damage"]))

    # ORIGINAL UNDAMAGED
    if num_damages['original'] > 0:
        dmg, att = no_damage(img)
        apply_damage(dmg, att)

    # QUADRANT
    if num_damages['quadrant'] > 0:
        quad_nums = rand.sample(range(1, 5), min(num_damages['quadrant'], 4))
        for n in quad_nums:
            dmg, att = remove_quadrant(img, n)
            apply_damage(dmg, att)
    
    # BIG HOLE
    if num_damages['big_hole'] > 0:
        angles = rand.sample(
            range(0, 360, 20), num_damages['big_hole'])
        for a in angles:
            dmg, att = remove_hole(img, a)
            apply_damage(dmg, att)
    
    # RANDOMISED BULLET HOLES
    b_conf = config['bullet_holes']
    if num_damages['bullet_holes'] > 0:
        hole_counts = rand.sample(
            range(b_conf['min_holes'], b_conf['max_holes'], 5), num_damages['quadrant'])
        for num_holes in hole_counts:
            dmg, att = bullet_holes(img, int(num_holes), b_conf['target'])
            apply_damage(dmg, att)
            if 0 < b_conf['target'] < 1.0: break

    # BEND
    bd_config = config['bend']
    if num_damages['bend'] > 0:
        # num_bg_bends = max(1, num_damages['bend'] // (len(backgrounds) or 1))
        num_bg_bends = max(1, num_damages['bend'])
        bend_angles = rand.sample(
            range(10, max(bd_config['max_bend'] + 5, 15), 5), num_bg_bends)
        
        for bend in bend_angles:
            axis = rand.randint(0, bd_config['max_axis'])
            if len(backgrounds) == 0:  # i.e. if detect_light_src == False
                dmg, att = bend_vertical(img, axis, bend, beta_diff=0)
                apply_damage(dmg, att)
            else:
                # Create different bent sign image for each background
                for bg in backgrounds:  # TODO Allen: "could be refactored so it looks a bit less ugly"
                    light_x, light_y = bg.light_coords
                    intensity = bg.light_intensity
                    fg_height, fg_width = img.shape[:2]
                    fg_x, fg_y, fg_size = SynthImage.gen_sign_coords(bg.shape[:2], (fg_height, fg_width))
                    
                    # Calculate beta difference
                    vec1 = np.array([math.cos(math.radians(90-axis)), math.sin(math.radians(90-axis))])
                    vec2 = np.array([fg_x - light_x, fg_y - light_y])
                    angle = math.acos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
                    beta_diff = math.copysign(1, vec2[0]) * math.sin(angle) * intensity * (bend / 90) * 4
                    
                    # Apply bending damage
                    dmg, att = bend_vertical(img, axis, bend, beta_diff=beta_diff)
                    
                    # Create SynthImage
                    dmg_path = os.path.join(
                        output_path, f"{class_num}_{att['damage_type']}_{att['tag']}_BG_{Path(bg.path).stem}.png")
                    cv.imwrite(dmg_path, dmg)
                    damaged_images.append(SynthImage(
                        dmg_path, int(class_num), att["damage_type"], att["tag"], float(att["damage_ratio"]), 
                        att["sector_damage"], bg_path=bg.path, fg_coords=(fg_x, fg_y), fg_size=fg_size))

    # GRAFFITI
    g_conf = config['graffiti']
    if num_damages['graffiti'] > 0:
        targets = np.linspace(g_conf['initial'], g_conf['final'], num_damages['graffiti'])
        for t in targets:
            dmg, att = graffiti(img, target=t, color=(0,0,0), solid=g_conf['solid'])
            apply_damage(dmg, att)

    # TINTED YELLOW
    # TODO: Utilise config file for max and min tint values with associated bounds checking in create_dataset.py
    if num_damages['tint_yellow'] > 0:
        tints = rand.sample(range(120, 240, 10), num_damages['tint_yellow'])
        for t in tints:         # ^ I chose this range pretty arbitrarily when reimplementing this damage type
            dmg, att = tint_yellow(img, tint=t)
            apply_damage(dmg, att)

    # GREY
    # TODO: Utilise config file for max and min beta values with associated bounds checking in create_dataset.py
    # FIXME: Completely oversaturates sign types which are already grey (i.e. STSDG sign 49)
    #        Try using gamma correction instead of convertScaleAbs (see manipulate.py)
    if num_damages['grey'] > 0:
        betas = rand.sample(range(100, 240, 5), num_damages['grey'])
        for b in betas:         # ^ This min beta is arbitrary, should be in config; I think 240 is good max though
            dmg, att = grey(img, beta=b)
            apply_damage(dmg, att)
    
    #       See reference image collection for inspiration
    # TODO: CRACKS (thin crack lines across the sign? textured global 'cracking'?)
    # TODO: MISSING SECTIONS (more generalised; missing polygon sections on edges of sign?)
    # TODO: DENTS and/or CURVED BENDING (as opposed to instantaneous bending over a straight line)

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
    # dmg = cv.bitwise_and(img, img, mask=dmg[:,:,3])

    # Assign labels
    att = attributes
    att["damage_type"]   = "no_damage"
    att["tag"]           = ""
    att["damage_ratio"]  = str(calc_damage(dmg, img, dmg_measure))  # This should be 0.0
    att["sector_damage"] = calc_damage_sectors(dmg, img, method=dmg_measure, num_sectors=num_sectors)

    return dmg, att

def tint_yellow(img, tint=180):
    dmg = validate_sign(img)
    hole = dmg[:,:,3].copy()
    att = attributes

    height, width, ch = dmg.shape

    # TODO: Perhaps there could be more variation in colour
    #       e.g. have tint1 and tint2, with each being tint +- x, a new variable
    yellow = np.zeros((height,width,ch), dtype=np.uint8)
    yellow[:,:] = (0,tint,tint,255)
    dmg = cv.bitwise_and(dmg, yellow)

    # Assign labels
    att = attributes
    att["damage_type"]   = "tint_yellow"
    att["tag"]           = str(int(tint))
    att["damage_ratio"]  = "{:.3f}".format(calc_damage(dmg, img, dmg_measure))
    att["sector_damage"] = calc_damage_sectors(dmg, img, method=dmg_measure, num_sectors=num_sectors)

    return dmg, att

def grey(img, beta=200):  # TODO: Imitates complete fading of colour in sign: perhaps rename?
    dmg = validate_sign(img)

    channels = cv.split(img)

    dmg = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)  # Convert to greyscale
    # Threshold the image to get a uniform saturation
    _, dmg = cv.threshold(dmg, 100, 255, cv.THRESH_BINARY)
    cv.convertScaleAbs(dmg, dmg, alpha=1, beta=beta)  # No change to contrast, scale brightness
    dmg = cv.cvtColor(dmg, cv.COLOR_GRAY2BGRA)  # Convert back to BGRA to add back the alpha channel
    dmg[:,:,3] = channels[3]

    # Assign labels
    att = attributes
    att["damage_type"]   = "grey"
    att["tag"]           = str(int(beta))
    att["damage_ratio"]  = "{:.3f}".format(calc_damage(dmg, img, dmg_measure))
    att["sector_damage"] = calc_damage_sectors(dmg, img, method=dmg_measure, num_sectors=num_sectors)

    return dmg, att

def remove_quadrant(img, quad_num=-1):
    """Make one random quandrant transparent."""
    dmg = validate_sign(img)
    quadrant = dmg[:,:,3].copy()

    height, width, _ = img.shape
    centre_x = int(round(width / 2))
    centre_y = int(round(height / 2))

    if quad_num == -1:
        quad_num = rand.randint(1, 4)
    # Remove the quadrant: -1 offset is necessary to avoid damaging part of a wrong quadrant
    if quad_num == 1:         # top-right          centre
        cv.rectangle(quadrant, (width, 0), (centre_x, centre_y-1), 0, thickness=-1)
    elif quad_num == 2:       # top-left           centre
        cv.rectangle(quadrant, (0, 0), (centre_x-1, centre_y-1), 0, thickness=-1)
    elif quad_num == 3:       # bottom-left        centre
        cv.rectangle(quadrant, (0, height), (centre_x-1, centre_y), 0, thickness=-1)
    elif quad_num == 4:       # bottom-right       centre
        cv.rectangle(quadrant, (width, height), (centre_x, centre_y), 0, thickness=-1)
    
    dmg = cv.bitwise_and(img, img, mask=quadrant)

    # Assign labels
    att = attributes
    att["damage_type"]   = "quadrant"
    att["tag"]           = str(quad_num)
    att["damage_ratio"]  = "{:.3f}".format(calc_damage(dmg, img, dmg_measure))  # This should be around 0.25
    att["sector_damage"] = calc_damage_sectors(dmg, img, method=dmg_measure, num_sectors=num_sectors)

    return dmg, att

def remove_hole(img, angle):
    """Remove a circle-shaped region from the edge of the sign."""
    dmg = validate_sign(img)
    hole = dmg[:,:,3].copy()
    att = attributes

    height, width, _ = dmg.shape
    centre_x = int(round(width / 2))
    centre_y = int(round(height / 2))

    radius = int(2 * height / 5)
    rad = -(angle * math.pi / 180)  # Radians
    x = centre_x + int(radius * math.cos(rad))  # x-coordinate of centre
    y = centre_y + int(radius * math.sin(rad))  # y-coordinate of centre

    cv.circle(hole, (x,y), radius, (0,0,0), -1)  # -1 to create a filled circle
    dmg = cv.bitwise_and(dmg, dmg, mask=hole)

    # Assign labels
    att = attributes
    att["damage_type"]   = "big_hole"
    att["tag"]           = str(int(angle))
    att["damage_ratio"]  = "{:.3f}".format(calc_damage(dmg, img, dmg_measure))
    att["sector_damage"] = calc_damage_sectors(dmg, img, method=dmg_measure, num_sectors=num_sectors)

    return dmg, att

def bullet_holes(img, num_holes=40, target=-1):
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
            if (size < 6):  # TODO: Make relative to image width
                cv.circle(painted, (x,y), size, (hl,hl,hl,255), -1)
            else:
                cv.circle(bullet_holes, (x,y), size, (0,0,0), -1)
                dmg = cv.bitwise_and(painted, painted, mask=bullet_holes)
            
            ratio = round(calc_damage(painted, img, dmg_measure), 3)
            num_holes += 1
    # Apply damage with a random number of holes within the specified min-max range
    else:
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
            if (size < 6):  # TODO: Make relative to image width
                cv.circle(painted, (x,y), size, (hl,hl,hl,255), -1)
            else:
                cv.circle(bullet_holes, (x,y), size, (0,0,0), -1)

        dmg = cv.bitwise_and(painted, painted, mask=bullet_holes)
        ratio = round(calc_damage(painted, img, dmg_measure), 3)

    # Assign labels
    att = attributes
    att["damage_type"]   = "bullet_holes"
    att["tag"]           = str(num_holes)
    att["damage_ratio"]  = "{:.3f}".format(ratio)
    att["sector_damage"] = calc_damage_sectors(painted, img, method=dmg_measure, num_sectors=num_sectors)

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

def draw_bezier(grft, x0, y0, x1, y1, x2, y2, thickness, color, solid):
    """Draw a single bezier curve of desired thickness and colour."""
    from skimage import morphology

    if solid:
        start = -(thickness // 2)
        end = thickness // 2
        for ii in range(start, end+1):
            # Draw curves, shifting vertically and adjusting x-coordinates to round the edges
            yy, xx = draw.bezier_curve(y0+ii, x0+abs(ii), y1+ii, x1, y2+ii, x2-abs(ii), weight=1, shape=grft.shape)
            
            grft[yy, xx] = 255  # Modify the colour of the pixels that belong to the curve
        # Dilate the image to fill in the gaps between the bezier lines
        # grft = morphology.dilation(grft, morphology.disk(radius=1))
    else:
        alpha = rand.randint(240, 255)  # Random level of transparency
        # skimage.draw.bezier_curve() only draws curves of thickness 1 pixel,
        # We layer multiple curves to produce a curve of desired thickness
        start = -(thickness // 2)
        end = thickness // 2
        for ii in range(start, end+1):
            # Draw curves, shifting vertically and adjusting x-coordinates to round the edges
            yy, xx = draw.bezier_curve(y0+ii, x0+abs(ii), y1+ii, x1, y2+ii, x2-abs(ii), weight=1, shape=grft.shape)
            
            grft[yy, xx] = color + (alpha,)  # Modify the colour of the pixels that belong to the curve
    return grft

def graffiti(img, target=0.2, color=(0,0,0), solid=True):
    """Apply graffiti damage to sign.
       :param initial: the first target level of obscurity (0-1)
       :param final: the level of obscurity to stop at (0-1)
       :returns: a list containing the damaged images, and a list with the corresponding attributes
    """
    from skimage import morphology

    validate_sign(img)
    height, width, _ = img.shape
    if solid:
        grft = np.zeros((height, width), dtype=np.uint8)  # New blank image for drawing the graffiti on.
    else:
        grft = np.zeros((height, width, 4), dtype=np.uint8)  # New blank image for drawing the graffiti on.

    ratio = 0.0
    x0, y0 = width//2, height//2  # Start drawing in the centre of the image
    
    # Keep drawing bezier curves until the obscurity hits the target
    while ratio < target:
        radius = width // 5  # Radius of max distance to the next point
        x1, y1, x2, y2 = calc_points(img, x0, y0, radius)
        thickness = int(round(width // 20))
        grft = draw_bezier(grft, x0, y0, x1, y1, x2, y2, thickness, color, solid)
        # Make the end point the starting point for the next curve
        x0, y0 = x2, y2
        if solid:
            grft_dilated = morphology.dilation(grft, morphology.disk(radius=1))
            ratio = round(calc_ratio(grft_dilated, img), 4)
        else:
            ratio = round(calc_ratio(grft, img), 4)
    # Add a copy to the list and continue layering more graffiti
    
    if solid:
        grft = grft_dilated
        grft = cv.cvtColor(grft, cv.COLOR_GRAY2BGRA)

    grft[:,:,3] = cv.bitwise_and(grft[:,:,3], img[:,:,3])  # Combine with original alpha to remove any sign spillover

    if solid:
        grft = cv.bitwise_and(img, img, mask=grft_dilated)
        alpha = cv.split(grft)[3]
        grft[alpha == 255] = color + (255,)

    # Apply a Gaussian blur to each image, to smooth the edges
    k = (int(round( width/30 )) // 2) * 2 + 1  # Kernel size must be odd
    grft = cv.GaussianBlur(grft, (k,k), 0)
    dmg = overlay(grft, img)
    
    # Assign labels
    att = attributes
    att["damage_type"]   = "graffiti"
    att["tag"]           = str(round(target, 3))
    att["damage_ratio"]  = "{:.3f}".format(calc_damage(dmg, img, dmg_measure))
    att["sector_damage"] = calc_damage_sectors(dmg, img, method=dmg_measure, num_sectors=num_sectors)
    return dmg, att


### The following three functions are for the 'bend' damage type ###

def combine(img1, img2, beta_diff=-20):
    """Combine the left half of img1 with right half of img2 and returns the result."""
    _,wd,_ = img1.shape
    result = img1.copy()
    right  = img2.copy()
    # Save the alpha data (to replace later), as convertScaleAbs() will affect the transparency
    result_alpha = cv.split(result)[3]
    right_alpha  = cv.split(right)[3]
    
    # Darken the entire image of the copy
    alpha = 1  # No change to contrast
    beta  = beta_diff / 2  # Decrease brightness
    cv.convertScaleAbs(result, result, alpha, beta)
    cv.convertScaleAbs(right, right, alpha, -beta)
    # Replace the alpha data
    result[:,:,3] = result_alpha
    right[:,:,3]  = right_alpha
    
    # Copy over the right half of img2 onto the darkened image
    result[:,wd//2:wd] = right[:,wd//2:wd]

    return result

def tilt(img, angle, dims):                                                                     
        ht, wd = dims
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
    """Apply perspective warp to tilt images and combine to produce bent signs. Bend along a specified axis by rotating
    the image by the axis angle before performing horizontal bending, and then rotating by the axis angle in the
    opposite direction.
       :param img: the image to use to produce damaged signs
       :param max_axis: the maximum bearing of the axis of rotation in x-y plane
       :params max_bend: the maximum degree of inwards bending
       :returns: a list of bent signs and a list of corresponding attributes
    """ 
    rot_img = imutils.rotate_bound(img, axis_angle)
    rot_img = remove_padding(rot_img)
    ht, wd,_ = rot_img.shape  # Retrieve image dimentions
    left, right = tilt(rot_img, bend_angle, (ht, wd))

    def apply_bend(left, right, tag):
        dmg = combine(left, right, beta_diff)
        dmg = imutils.rotate_bound(dmg, -axis_angle)
        dmg = remove_padding(dmg)
        
        # Pad images to the same size to perform pixel comparison
        pad_h = max(img.shape[0], dmg.shape[0])
        pad_w = max(img.shape[1], dmg.shape[1])
        original = pad(img, pad_h, pad_w)
        dmg = pad(dmg, pad_h, pad_w)
        
        att = attributes
        att["damage_type"]   = "bend"
        att["tag"]           = "{}_{}".format(axis_angle, bend_angle)
        att["damage_ratio"]  = "{:.3f}".format(calc_damage(dmg, original, dmg_measure))
        att["sector_damage"] = calc_damage_sectors(dmg, original, method=dmg_measure, num_sectors=num_sectors)
        return dmg, att
        
    opt = rand.choice(["left", "right", "both"])
    if opt == "left":
        dmg, att = apply_bend(rot_img, right, 'right')
    elif opt == "right":
        dmg, att = apply_bend(left, rot_img, 'left')
    else:
        dmg, att = apply_bend(left, right, 'both')
    return dmg, att