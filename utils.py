"""Utility functions for manipulating images."""
# Coauthors: Kristian Rados, Seana Dale, Jack Downes


import cv2
import numpy as np
import os

def load_paths(directory):
    """Returns a list with the paths of all files in the directory"""
    paths = []
    for filename in os.listdir(directory): # Retrieve names of files in directory
        # Concatenate filename with directory path, ignoring hidden files
        path = os.path.join(directory, filename)
        if not filename.startswith('.'):
            paths.append(path)
    return paths

def load_files(directory):
    """Returns a list with the paths of all non-directory files in the directory"""
    paths = []
    for filename in os.listdir(directory): # Retrieve names of files in directory
        # Concatenate filename with directory path, ignoring hidden files and directories
        path = os.path.join(directory, filename)
        if os.path.isfile(path) and not filename.startswith('.'):
            paths.append(path)
    return paths

def dir_split(path):
    """Imitates the functionality of path.split('/') while using os.path.split"""
    # Source: https://stackoverflow.com/a/3167684/12350950
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break
    folders.reverse()
    return folders


def match_height(img, new_height):
    old_height, old_width, _ = img.shape  # Discard channel
    new_width = int(round( new_height / old_height * old_width ))
    img = cv2.resize(img, (new_height, new_width))
    return img

def match_width(img, new_width):
    old_width, old_height, _ = img.shape
    new_height = int(round( new_width / old_width * old_height ))
    img = cv2.resize(img, (new_width, new_height))
    return img

def resize(img1, img2):
    """Resize img1 to ensure it is not larger than img2."""
    # Retrieve dimentions of both images
    height1, width1, _ = img1.shape  # Discard channel
    height2, width2, _ = img2.shape
    
    # Resize if img1 is taller than img2
    if height1 > height2:
        img1 = match_height(img1, height2)
        height1, width1, _ = img1.shape  # Re-retrieve dimentions
    # Or wider
    if width1 > width2:
        img1 = match_width(img1, width2)
    
    return img1, img2


def overlay(fg, bg, x1=-1, y1=-1):
    """Overlay foreground image on background image, keeping transparency.
       :param x1, y1: top-left coordinates to place foreground image \n
       Foreground image will be centred on background if x1 or y1 is omitted. \n
       Images may be RGB or RGBA.
    """
    # If the background doesn't have an alpha channel, add one
    if len(cv2.split(bg)) == 3:
        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2RGBA)
        bg[:, :, 3] = 255  # Keep it opaque
    # Same for foreground
    if len(cv2.split(fg)) == 3:
        fg = cv2.cvtColor(fg, cv2.COLOR_RGB2RGBA)
        fg[:, :, 3] = 255
    
    # Make a copy of the background for making changes
    new_img = bg.copy()

    # Retrieve image dimentions
    height_FG, width_FG, _ = fg.shape  # Discard channel
    height_BG, width_BG, _ = bg.shape

    # If either of the coordinates were omitted, calculate start/end positions
    # using the difference in image size, centring foreground on background
    if x1 == -1 or y1 == -1:
        # Start coordinates 
        x1 = (width_BG - width_FG) // 2    # Floor division to truncate as
        y1 = (height_BG - height_FG) // 2  # coordinates don't need to be exact
    # End coordinates
    x2 = x1 + width_FG
    y2 = y1 + height_FG

    ### Start of code from fireant ###
    # https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
    # Retrieve an array of alpha values from the foreground image
    # Divide by 255 to get values between 0.0 and 1.0
    alpha = fg[:,:,3] / 255.0
    beta = 1.0 - alpha

    # Loop over BGR channels (but not alpha)
    for ch in range(0, 3):
        new_img[y1:y2, x1:x2, ch] = (alpha * fg[:, :, ch] +
                                     beta * new_img[y1:y2, x1:x2, ch])
    ### End of code from fireant ###

    return new_img


def count_pixels(img):
    """Return the number of non-transparent pixels in the imported image."""
    sum = 0  # Initialise to 0 in case there is no alpha channel
    split = cv2.split(img)
    if len(split) == 4:  # Only proceed if the image has an alpha channel
        alpha = split[3]
        # Loop through alpha channel
        for ii in range(0, len(alpha)):
            for jj in range(0, len(alpha[0])):
                sum += alpha[ii][jj] / 255  # Divide by 255 to weight the transparency
    return sum

def calc_ratio(fg, bg):
    """Calculate the ratio of obscurity in first image compared to second image.
       :param fg: the foreground (if overlaying) or smaller image
       :param bg: the background or larger image
       :returns: the ratio, a number between 0 and 1"""
    # Compare the number of non-transparent pixels in the two images
    fg_pixels = count_pixels(fg)
    bg_pixels = count_pixels(bg)
    ratio = fg_pixels / bg_pixels
    # If bg had less pixels than fg, flip the fraction to ensure ratio is 0-1
    if ratio > 1:
        ratio = 1 / ratio

    return ratio


def count_diff_pixels(new, original):
    """Count how many opaque pixels are different between the two imported images. \n
    Based on count_pixels()."""
    count = 0
    split = cv2.split(original)
    if len(split) == 4:
        alpha = split[3]
        for ii in range(0, len(new)):
            for jj in range(0, len(new[0])):
                if alpha[ii][jj] > 0:  # The pixel in the original image must have been opaque
                    if np.any(new[ii][jj] != original[ii][jj]):  # Check for any changes in the pixel
                        count += 1
    return count

def calc_quadrant_diff(new, original):
    """Calculate the ratio of changed opaque pixels between two versions of the same image for each quadrant."""
    height, width, _ = original.shape
    centre_x = int(round( width / 2 ))
    centre_y = int(round( height / 2 ))

    # Divide both images into quadrants
    new_I   = new[0:centre_y, centre_x:width]
    new_II  = new[0:centre_y, 0:centre_x]
    new_III = new[centre_y:height, 0:centre_x]
    new_IV  = new[centre_y:height, centre_x:width]
    original_I   = original[0:centre_y, centre_x:width]
    original_II  = original[0:centre_y, 0:centre_x]
    original_III = original[centre_y:height, 0:centre_x]
    original_IV  = original[centre_y:height, centre_x:width]

    ratio_I   = count_diff_pixels(new_I, original_I) / count_pixels(original_I)
    ratio_II  = count_diff_pixels(new_II, original_II) / count_pixels(original_II)
    ratio_III = count_diff_pixels(new_III, original_III) / count_pixels(original_III)
    ratio_IV  = count_diff_pixels(new_IV, original_IV) / count_pixels(original_IV)

    return [ratio_I, ratio_II, ratio_III, ratio_IV]


def append_labels(image_path, axes, class_id, dmg, labels_path):
    """Append the label for an image to the labels/annotations file.
    
    Arguments:
    image_path  -- file path to the image (with extension)
    axes        -- list of integer bounding box axes [left, right, top, bottom]
    class_id    -- integer class number
    dmg         -- list of float damage values for each quadrant [I, II, III, IV]
    labels_path -- file path to labels/annotations file
    """
    file = open(labels_path, "a")
    
    if True:
        file.write("{0} {1},{2},{3},{4},{5},{6},{7},{8},{9}\n" \
            .format(image_path, axes[0],axes[2],axes[1],axes[3], class_id, dmg[0],dmg[1],dmg[2],dmg[3]))
    else:  # TODO: Only standard detection labels if damage info option set to false
        file.write("{0} {1},{2},{3},{4},{5}\n" \
            .format(image_path, axes[0],axes[2],axes[1],axes[3], class_id))
    file.close()