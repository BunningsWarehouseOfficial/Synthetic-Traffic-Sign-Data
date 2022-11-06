"""Module of functions for generating a synthetic manipulated sign dataset."""

import os
from datetime import datetime
import random
import yaml

import imutils
import cv2
import numpy as np

from utils import load_paths, dir_split, overlay, overlay_new, adjust_contrast_brightness, random_noise_method
from synth_image import SynthImage

# Open config file
with open("config.yaml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)


def _has_opaque_pixel(line):
    """Checks if a line of pixels contains a pixel above a transparency threshold."""
    opaque = False
    for pixel in line:
        # Check if pixel is opaque
        if pixel[3] > 200:
            opaque = True
            break  # Stop searching if one is found
    return opaque

def bounding_axes(img):  # TODO: Attempt speedup by making this function a regionprops wrapper
    """Returns the bounding axes of an image with a transparent background."""
    # Top axis
    y_top = 0
    for row in img:  # Iterate through each row of pixels, starting at top-left
        if _has_opaque_pixel(row) is False:  # Check if the row has an opaque pixel
            y_top += 1  # If not, move to the next row
        else:
            break  # If so, break, leaving y_top as the bounding axis

    # Bottom axis
    height = img.shape[0]
    y_bottom = height - 1
    for row in reversed(img):  # Iterate from the bottom row up
        if _has_opaque_pixel(row) is False:
            y_bottom -= 1
        else:
            break

    # Left axis
    # Rotate 90 degrees to iterate through what were originally columns
    r_img = imutils.rotate_bound(img, 90)
    x_left = 0
    for column in r_img:
        if _has_opaque_pixel(column) is False:
            x_left += 1
        else:
            break

    # Right axis
    r_height = r_img.shape[0]
    x_right = r_height - 1
    for column in reversed(r_img):
        if _has_opaque_pixel(column) is False:
            x_right -= 1
        else:
            break

    ## For debug
    # img[y_top, :] = (255, 0, 0, 255)
    # img[y_bottom, :] = (255, 0, 0, 255)
    # img[:, x_left] = (255, 0, 0, 255)
    # img[:, x_right] = (255, 0, 0, 255)
    # cv2.imwrite(image_dir, img)
    ##

    ##
    # For debug (place outside this function)
    # for img in load_paths("Traffic_Signs_Templates/4_Transformed_Images/0/0_ORIGINAL"):
    #     bounding_axes(img)  
    ##  

    return [x_left, x_right, y_top, y_bottom]


def _augment_image(img):
    """Augment an image."""
    a_config = config['augments']

    # Brightness/contrast variation
    alpha = random.uniform(a_config['min_alpha'], a_config['max_alpha'])
    beta = random.randint(a_config['min_beta'], a_config['max_beta'])
    img = adjust_contrast_brightness(img, alpha, beta)
    return img

def _augment_fg_image(img):
    """Apply augmentations specific to the foreground images."""
    a_config = config['augments']

    # Apply non-specific augmentations
    img = _augment_image(img)

    # Speckle/Guassian/Poisson noise
    noise_types = [n_type for n_type in a_config['noise_types'] if a_config['noise_types'][n_type] is True]
    img = random_noise_method(img, noise_types, a_config['noise_vars'])
    return img

def _augment_final_image(img):
    """Apply augmentations specific to the final image with all signs."""
    a_config = config['augments']

    # Apply non-specific augmentations
    img = _augment_image(img)

    # Apply normalizing augmentations
    if a_config['non_local_means_denoising'] is True:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 48, 7, 5)
    # Default (sigmaX=0) calculated sigma for kernel size 3 is 0.8
    img = cv2.GaussianBlur(img, (a_config['gaussian_kernel'], a_config['gaussian_kernel']), a_config['gaussian_sigma'])

    ## DEBUG
    # cv2.imshow("final", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ##
    return img

def new_data(synth_image_set, online=False):
    """Blends a set of synthetic signs with the corresponding background."""
    bg_path = synth_image_set[0].bg_path
    bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
    assert bg is not None, "Background image not found"

    def get_fg(synth_image, online):
        if online is True:
            fg = synth_image.fg_image
        else:
            fg_path = synth_image.fg_path
            fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        assert fg is not None, "Foreground image not found"
        return _augment_fg_image(fg)

    bboxes = []
    total = 0   # Signs placed so far
    nb_signs_in_img = len(synth_image_set)  # Total number of signs attempting to place
    fail = 0    # Attempts to place signs
    while total < nb_signs_in_img and fail < 40:
        synth_image = synth_image_set[total]
        fg = get_fg(synth_image, online)
        
        if synth_image.fg_coords is not None and synth_image.fg_size is not None:
            x, y = synth_image.fg_coords
            new_size = synth_image.fg_size
        else:
            x, y, new_size = SynthImage.gen_sign_coords(
                bg.shape[:2],
                fg.shape[:2],
                config['sign_placement']['min_ratio'],
                config['sign_placement']['max_ratio'],
                config['sign_placement']['middle_third'],
            )

        bg, bbox = overlay_new(fg, bg, new_size, bboxes, x, y)
        if bbox is not None:
            bboxes.append(bbox)
            total += 1
            fg = cv2.resize(fg, (new_size, new_size))
            axes = bounding_axes(fg)  # Retrieve bounding axes of the sign image
            axes[0] += x  # Adjusting bounding axis to make it relative to the whole bg image
            axes[1] += x
            axes[2] += y
            axes[3] += y
            synth_image.bounding_axes = axes
            do_place_below = config['vertical_grid_placement']
        else:
            fail += 1

        # Place signs side-by-side in grid pattern
        # TODO: Add horizontal placement, currently only does vertical placement
        while do_place_below and total < nb_signs_in_img and fail < 40:
            do_place_below = np.random.choice([True]*5 + [False]) and total < nb_signs_in_img
            if not (do_place_below and total < nb_signs_in_img and bbox):
                break
            # position = (bbox['xmin'], bbox['ymax'])
            synth_image = synth_image_set[total]
            fg = get_fg(synth_image, online)

            if (bbox[0] + new_size >= bg.shape[1]) or (bbox[3] + new_size >= bg.shape[0]):
                break

            # Uses same new_size as previous sign
            bg, bbox = overlay_new(fg, bg, new_size, bboxes, bbox[0], bbox[3])
            if bbox is not None:
                bboxes.append(bbox)
                total += 1
                fg = cv2.resize(fg, (new_size, new_size))
                axes = bounding_axes(fg)  # Retrieve bounding axes of the sign image
                axes[0] += x  # Adjusting bounding axis to make it relative to the whole bg image
                axes[1] += x
                axes[2] += y
                axes[3] += y
                synth_image.bounding_axes = axes
            else:
                fail += 1

    image = _augment_final_image(bg)
    return image, total


# TODO: These two functions should be one function always include background using classes?
# List of paths for all SGTSD relevant files using exposure_manipulation
def paths_list(imgs_directory, bg_directory):
    directories = []
    for places in load_paths(imgs_directory):  # List of places: originally either UK_rural or UK_urban
        for imgs in load_paths(places):  # Folder for each bg image: eg. IMG_0
            dr = dir_split(imgs)
            bg = os.path.join(bg_directory, dr[-2], dr[-1] + ".png")  # Retrieving relevant bg image
            for signs in load_paths(imgs):  # Folder for each sign type: eg. SIGN_9
                for dmgs in load_paths(signs):  # Folder for each damage type: eg. 0_HOLES
                    for png in load_paths(dmgs):
                        directories.append([png, bg])
    return directories  # Directory for every single FILE and it's relevant bg FILE

# List of paths for all SGTSD relevant files using fade_manipulation; backgrounds are assigned to 
def assigned_paths_list(imgs_directory, bg_directory):  # TODO: Is this the same as above?
    directories = []
    for places in load_paths(bg_directory):  # Folder for each place: eg. GTSDB
        for imgs in load_paths(places):  # Iterate through each b.g. image: eg. IMG_0
            for signs in load_paths(imgs_directory):  # Folder for each sign type: eg. SIGN_9
                for dmgs in load_paths(signs):  # Folder for each damage type: eg. 9_HOLES
                    for png in load_paths(dmgs):
                        directories.append([png, imgs])
    return directories  # Directory for every single FILE and its relevant bg FILE

# Paths of images needed to generate examples for 'sign' with damage 'dmg'
def list_for_sign_x(sign, dmg, directories):
    l = []
    for elements in directories:
        foreground = dir_split(elements[0])
        if (foreground[-2] == sign + dmg):  # Eg. if (9_YELLOW == 4_ORIGINAL)
            l.append(elements)
    return l  # Directory for every single sign and its relevant background image