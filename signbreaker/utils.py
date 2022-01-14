"""Utility functions for manipulating images."""
# Coauthors: Kristian Rados, Seana Dale, Jack Downes

import cv2
import numpy as np
import os
from PIL import Image, ImageOps

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


def scale_image(image_path, width):
    """Rescales and pads the source image with whitespace to be a perfect square of fixed width"""
    # https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    # Resize the image
    img = Image.open(image_path)
    old_size = img.size  # old_size is in (width, height) format
    ratio = float(width) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.BICUBIC)

    # Pad the image with whitespace
    delta_w = width - new_size[0]
    delta_h = width - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    new_img = ImageOps.expand(img, padding, fill=(255, 255, 255, 0))
    
    return new_img

def create_alpha(img, alpha_channel):
    """Returns an alpha channel that matches the white background \n
    Based on Alexandros Stergiou's find_borders() function"""

    # Read and decode the image contents for pixel access
    pix = img.load()

    # Note PIL indexes [x,y] while OpenCV indexes [y,x]
    # alpha_channel must be indexed [y,x] to merge with OpenCV channels later

    min = 200
    width, height = img.size
    # Loop through each row of the image
    for y in range(0, height):
        # First loop left to right
        for x in range(0, width, 1):
            # Retrieve a tuple with RGB values for this pixel
            rgb = pix[x,y]
            # Make transparent if the pixel is white (or light enough)
            if rgb[0] >= min and rgb[1] >= min and rgb[2] >= min:
                alpha_channel[y,x] = 0
            # If pixel is not white then we've hit the sign so break out of loop
            else:
                break

        # Then loop backwards, right to left
        for x in range(width-1, -1, -1):
            rgb = pix[x,y]
            if rgb[0] >= min and rgb[1] >= min and rgb[2] >= min:
                alpha_channel[y,x] = 0
            else:
                break

    return alpha_channel

def delete_background(image_path, save_path):
    """Deletes the white background from the original sign
    Based on Alexandros Stergiou's manipulate_images() function"""
    # Open the image using PIL (don't read contents yet)
    img = Image.open(image_path)
    img = img.convert('RGB')  # Does this have any effect??

    # Open the image again using OpenCV and split into its channels
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    channels = cv2.split(image)

    # Create a fully opaque alpha channel, same dimentions and dtype as the image
    # create_alpha() modifies it to make the white background transparent
    alpha_channel = np.ones(channels[0].shape, dtype=channels[0].dtype) * 255
    alpha_channel = create_alpha(img, alpha_channel)

    # Merge alpha channel into original image
    image_RGBA = cv2.merge((channels[0], channels[1], channels[2], alpha_channel))

    cv2.imwrite(save_path, image_RGBA)

    img.close()

def to_png(directory):
    """Convert all files in 'directory' to PNG images"""
    for files in load_paths(directory):
        title, extension = files.split('.')
        img = Image.open(files).convert('RGBA')
        if (not extension == "png"):
            os.remove(files)
        img.save(title + ".png")

def png_to_jpeg(filepath):
    sep = os.sep

    dirs = dir_split(filepath)
    title,extension = dirs[-1].split('.')
    del dirs[-1]
    string = sep.join(dirs)
    string = os.path.join(string, title + ".jpg")
    png = Image.open(filepath)
    png.load()  # Required for png.split()
    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[-1])  # 3 is the alpha channel
    background.save(string, 'JPEG', quality=100)
    os.remove(filepath)


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


def add_pole(filename, color):
    """Adds a pole to the imported sign"""
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    # Retrieve image dimentions and calculate the new height
    height, width, _ = img.shape  # Discard channel
    new_height = height * 3

    # Create a new blank image with dtype=uint8 (to hold numbers 0-255)
    # Initialise values to 0 so that the extended image is fully transparent
    new_img = np.zeros((new_height, width, 4), dtype=np.uint8)
    # Copy over the original image onto the top portion
    new_img[0:height, :] = img

    # Calculate coordinates
    midpt = width // 2
    offset = width // 40
    x1, x2 = midpt - offset, midpt + offset
    y1, y2 = height, new_height
    # Draw the pole
    cv2.rectangle(new_img, (x1, y1), (x2, y2), color, cv2.FILLED)

    # Colour any transparent pixels between the sign and the rectangle
    lim = 50  # Max alpha value for the pixel to be considered "transparent"
    margin = int(round( height * 0.9 ))  # Loop over bottom 10% of the sign
    for y in range(margin, height):
        for x in range(x1, x2+1):  # Loop to x2 inclusive
            if new_img[y,x,3] < lim:
                new_img[y,x] = color

    return new_img


def overlay(fg, bg, x1=-1, y1=-1):
    """Overlay foreground image on background image, keeping transparency.

    Foreground iamge will be centred on background if x1 or y1 are omitted.
    Images may be RGB or RGBA.

    Arguments:
    x1, y1 -- top-left coordinates for where to place foreground image
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
    """Return the number of non-transparent pixels in the imported image weighted according
    to the transparency of each pixel."""
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
    """Calculate the ratio of obscurity in the first image as compared to the second image.

    Arguments:
    fg -- the foreground (if overlaying) or smaller image
    bg -- the background or larger image

    Returns:
    ratio -- a float between 0 and 1
    """
    # Compare the number of non-transparent pixels in the two images
    fg_pixels = count_pixels(fg)
    bg_pixels = count_pixels(bg)
    ratio = fg_pixels / bg_pixels
    # If bg had less pixels than fg, flip the fraction to ensure ratio is 0-1
    if ratio > 1:
        ratio = 1 / ratio

    return ratio


def count_diff_pixels(new, original):
    """Count how many opaque pixels have changed in any way between the two imported images. Based on count_pixels()."""
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


def count_damaged_pixels(new, original):
        #"""Count how many opaque pixels are different between the two imported images. Based on count_pixels()."""
    """Count how many pixels are different between the two imported images weighted by the transparency difference of
    each pixel first and the colour difference second. Based on count_pixels().""" ##
    
    ## For debug visualisations
    # copy = new.copy()
    ##

    split_new      = len(cv2.split(new))
    split_original = len(cv2.split(original))
    if split_new == 4 and split_original == 4:
        sum = 0
        for ii in range(0, len(new)):
            for jj in range(0, len(new[0])):
                if original[ii][jj][3] > 0:  # The pixel in the original image must have been opaque
                    # Colour diff is weighted by alpha diff, as it's more important
                    alpha_diff_ratio = abs(original[ii][jj][3] - new[ii][jj][3]) / 255
                    assert alpha_diff_ratio >= 0.0 and alpha_diff_ratio <= 1.0  # Will mostly be either 0.0 or 1.0

                    colour_diffs = [abs(int(original[ii][jj][x]) - int(new[ii][jj][x])) for x in range(0,3)]
                    cumulative_colour_diff = np.sum(colour_diffs)
                    assert cumulative_colour_diff >= 0 and cumulative_colour_diff <= 255*3

                    # Limited to 255 (somewhat arbitrary choice), otherwise only exact inverse would score as diff 1
                    colour_diff_max = 255
                    colour_diff_ratio = min(cumulative_colour_diff, colour_diff_max) / colour_diff_max
                    assert colour_diff_ratio >= 0.0 and colour_diff_ratio <= 1.0

                    damage_ratio = min(alpha_diff_ratio + colour_diff_ratio, 1.0)
                    sum += damage_ratio

                    ## For debug visualisations
                    # if np.any(new[ii][jj] != original[ii][jj]):  # Check for any changes in the pixel
                    #     copy[ii][jj] = (0,255*damage_ratio,0,255)
                    ##
    else:
        raise TypeError(f"The two images need 4 channels, but original has {split_new} and new has {split_original}")
    
    ## For debug visualisations
    # cv2.imshow("new", new) ##
    # cv2.waitKey(0) ##for debug visual
    # cv2.destroyAllWindows() ##for debug visual
    # cv2.imshow("count_damaged_pixels", copy) ##for debug visual
    # cv2.waitKey(0) ##for debug visual
    # cv2.destroyAllWindows() ##for debug visual
    ##

    return sum

def calc_damage(new, original):
    """Calculate the ratio of damaged pixels between two versions of the same image."""
    ## For debug
    # dmg = count_damaged_pixels(new, original)
    # total = count_pixels(original)
    # print(f"damage = count_damaged_pixels / count_pixels = {dmg} / {total} = {dmg / total}")
    # return dmg / total
    ##
    return count_damaged_pixels(new, original) / count_pixels(original)

def calc_damage_quadrants(new, original):
    """Calculate the ratio of damaged pixels between two versions of the same image for each quadrant."""
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

    ratio_I   = count_damaged_pixels(new_I, original_I) / count_pixels(original_I)
    ratio_II  = count_damaged_pixels(new_II, original_II) / count_pixels(original_II)
    ratio_III = count_damaged_pixels(new_III, original_III) / count_pixels(original_III)
    ratio_IV  = count_damaged_pixels(new_IV, original_IV) / count_pixels(original_IV)

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