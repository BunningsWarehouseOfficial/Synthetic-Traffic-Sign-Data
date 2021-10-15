def main():
    import os
    import ntpath
    import shutil
    import sys
    import argparse
    from datetime import datetime
    import random

    import numpy as np
    import cv2
    from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageFilter, ImageStat, ImageEnhance, ImageFile
    from skimage import io, color, exposure
    import matplotlib
    import matplotlib.pyplot as plt
    import glob
    import math

    from damage import no_damage, remove_quadrant, remove_hole, bullet_holes, graffiti, bend_vertical, damage_image
    from utils import load_paths, load_files, scale_image, create_alpha, delete_background, to_png, dir_split, png_to_jpeg
    import manipulate
    import generate
    from synth_image import SynthImage

    #TODO: Add an example download link for the GTSDB dataset in .png form

    # Open and validate config file
    import yaml
    with open("config.yaml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    #TODO: Input validation of config file
    #TODO: Add a "Using `config.yaml` configuration file." so that the user is aware of it

    # Directory names excluded from config.yaml to make use of .gitignore simpler
    base_dir        = "Sign_Templates"
    input_dir       = os.path.join(base_dir, "1_Input")
    processed_dir   = os.path.join(base_dir, "2_Processed")
    damaged_dir     = os.path.join(base_dir, "3_Damaged")
    transformed_dir = os.path.join(base_dir, "4_Transformed")
    manipulated_dir = os.path.join(base_dir, "5_Manipulated")
    bg_dir    = "Backgrounds"
    final_dir = "SGTS_Dataset"  # SGTSD stands for "Synthetically Generated Traffic Sign Dataset"

    # Create the output directories if they don't exist already
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    input_error = (f"Error: {input_dir} must be populated with template signs to proceed. A link to example "
                    "data can be found in the README.\n")
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
        raise ValueError(input_error)
    if len(load_paths(input_dir)) == 0:
        raise ValueError(input_error)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.mkdir(processed_dir)


    ### IMAGE PREPROCESSING ###
    # Rescale images and make white backgrounds transparent
    paths = load_files(input_dir)
    for path in paths:
        img = scale_image(path, config['sign_width']) # Rescale the image

        # Remove the extension and save as a png
        _, filename = ntpath.split(path)
        name, _ = filename.rsplit('.', 1)
        save_path = os.path.join(processed_dir, name) + ".png"
        img.save(save_path)

        delete_background(save_path, save_path) # Overwrite the newly rescaled image


    ### APPLYING DAMAGE ###
    # Remove any old output and recreate the output directory
    # reusable = config['reuse_damage']  #TODO: Implement optional damage reuse or any dir reuse (check downstream first)
    # if not reusable:
    #     shutil.rmtree(damaged_dir)
    if os.path.exists(damaged_dir):
        shutil.rmtree(damaged_dir) ##
    #if not os.path.exists(damaged_dir):
    os.mkdir(damaged_dir)
    damaged_data = []

    ii = 0
    processed = load_files(processed_dir)
    for image_path in processed:
        print(f"Damaging signs: {float(ii) / float(len(processed)):06.2%}", end='\r')
        damaged_data.append(damage_image(image_path, damaged_dir, config['damage_types']))
        ii += 1
    print(f"Damaging signs: 100.0%\r\n")
    damaged_data = [cell for row in damaged_data for cell in row]  # Flatten the list
    # else:
    #     print("Reusing pre-existing damaged signs")


    ### APPLYING TRANSFORMATIONS ###
    if os.path.exists(transformed_dir):
        shutil.rmtree(transformed_dir)
    os.mkdir(transformed_dir)

    transformed_data = []
    for damaged in damaged_data:
        save_dir = os.path.join(transformed_dir, str(damaged.class_num))
        transformed_data.append(manipulate.img_transform(damaged, save_dir, config['num_transform']))
        del damaged  # Clear memory that will no longer be needed as we go
    del damaged_data
    transformed_data = [cell for row in transformed_data for cell in row]  # Flatten the list


    ### MANIPULATING EXPOSURE/FADE ###
    ImageFile.LOAD_TRUNCATED_IMAGES = True  #TODO: Is this line needed?
    for bg_folders in load_paths(bg_dir):
        to_png(bg_folders)

    #TODO: This mess of a loop could probably be cleaned up?
    for dirs in load_paths(bg_dir):
        if config['man_method'] == 'exposure':
            for background in load_paths(dirs):
                iniitial, subd, element = dir_split(background)
                title, extension = element.split('.')

                for signp in load_paths(transformed_dir):
                    for sign in load_paths(signp):
                        d, s, f, e = dir_split(sign)  # Eg. s=4_Transformed, f=9, e=9_BIG_HOLE

                        #if (not os.path.exists(exp_dir + subd + sep + title + sep + "SIGN_" + f + sep + e)):
                        sign_dir = os.path.join(manipulated_dir, subd, title, "SIGN_" + f, e)
                        if not os.path.exists(sign_dir):
                            os.makedirs(sign_dir)
                            #os.makedirs(exp_dir + subd + sep + title + sep + "SIGN_" + f + sep + e)
        else:
            for signp in load_paths(transformed_dir):
                for sign in load_paths(signp):
                    d,s,f,e = dir_split(sign)

                    sign_dir = os.path.join(manipulated_dir, "SIGN_" + f, e)
                    if (not os.path.exists(sign_dir)):
                        os.makedirs(sign_dir)

    signs_paths = []
    for p in load_paths(transformed_dir):
        for d in load_paths(p):
            signs_paths += load_paths(d)

    background_paths = []  # Load the paths of the background images into a single list
    for subfolder in load_paths(bg_dir):
        background_paths += load_paths(subfolder)

    #TODO: Can do checks for damage type in below functions to avoid funky results cancelling manipulation for just those types
    
    if config['man_method'] == 'exposure':
        manipulated_data = manipulate.exposure_manipulation(transformed_data, background_paths, manipulated_dir)
    else:
        #TODO: Still need to do encapsulation here
        None
        #manipulated_data = manipulate.fade_manipulation(transformed_data, background_paths, manipulated_dir)

    if config['man_method'] == 'exposure':
        manipulate.find_useful_signs(manipulated_data, manipulated_dir, damaged_dir) #TODO: MAKE SURE TO DO THIS, DELETING SYNTH_IMAGE OBJECTS ALONG THE WAY

    # Delete SynthImage objects for any signs that were removed
    manipulated_data[:] = [x for x in manipulated_data if os.path.exists(x.fg_path)]


    ### GENERATING FINAL DATA ###
    images_dir = os.path.join(final_dir, "Images")
    labels_path = os.path.join(final_dir, "labels.txt")
    about_path = os.path.join(final_dir, "generated_images_about.txt")

    # Clean and recreate the parent images directory
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    os.makedirs(images_dir)
        
    total_gen = len(manipulated_data)
    print(f"Files to be generated: {total_gen}")

    ii = 0
    with open(labels_path, "w") as labels_file:
        for synth_image in manipulated_data:
            print(f"Generating files: {float(ii) / float(total_gen):06.2%}", end='\r')
            
            c_num = synth_image.class_num
            d_type = synth_image.damage_type
            class_dir = os.path.join(images_dir, f"{c_num}", f"{c_num}_{d_type}")
            # Create the directory for each class+damage combination if it doesn't already exist
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            
            fg_path =  os.path.join(class_dir, f"{c_num}_{d_type}_{ii}")
            temp_fg_path  = fg_path + ".png"
            final_fg_path = fg_path + ".jpg"  # It is assumed that the final .jpg -> .png conversion step is executed

            image = generate.new_data(synth_image, labels_file)
            cv2.imwrite(temp_fg_path, image)
            ii += 1
        print(f"Generating files: 100.0%\r\n")

    #TODO: Is this really necessary? Can't we save to JPEG directly?
    ii = 0
    class_dirs = load_paths(images_dir)
    for class_dir in class_dirs:
        for damage_dir in load_paths(class_dir):
            for image in load_paths(damage_dir):
                print(f"Converting to JPEG: {float(ii) / float(total_gen):06.2%}", end='\r')
                if image.endswith("png"):
                    png_to_jpeg(image)
                ii += 1  
    print(f"Converting to JPEG: 100.0%\r\n")

    string = '''
    -------------------------------------
    BREAKDOWN OF FILES GENERATED BY CLASS
    -------------------------------------\n
    '''
    for class_dir in load_paths(images_dir):
        c_total = 0
        for damage_dir in load_paths(class_dir):
            c_total += len(load_paths(damage_dir))
        _, c_num = ntpath.split(class_dir)
        c_str = f"Generated {c_total} examples for sign class {c_num}\n"
        string += c_str
    string += f"\nTOTAL: {total_gen}\n"
    string += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    string += "-------------------------------------"

    with open(about_path, "w") as text_file:
        text_file.write(string)


    ### CREATE DATASET README ### #TODO: Really necessary?
    content = '''
    -----------------------------------------------
    |                     -*-                     |
    |Synthetically Generated Traffic Sign Dataset |
    |                     -*-                     |
    -----------------------------------------------

    This directory contains the generated training
    set to be used for training a Convolutional
    Neural Network (CNN).

    It may be used for any detector desired and it
    is not limited to a specific set of traffic
    sign templates.
    

    ----------------------------------------------
    Content
    ----------------------------------------------

    The number of examples is based on the number:
    ->of traffic signs that were used as templates
    ->of damage types applied to images
    ->of transformations in image manipulations
    ->of the brightness variation values used
    ->of blending procedures


    ----------------------------------------------
    Image format and naming
    ----------------------------------------------

    The images created are of "jpg" format
    with RGBA channels.

    XXX/XXX_YYY/XXX_YYY_ZZZ.jpg

    (X) is used to distinguish the sign class, (Y)
    is used to distinguish the damage type and (Z)
    is used to indicate the example number.


    ----------------------------------------------
    Additional information
    ----------------------------------------------

    Contact Email:
    Original Code:
        asterga@essex.ac.uk

    Adapted Damage Code:
        kristian.rados@student.curtin.edu.au
        seana.dale@student.curtin.edu.au


    ----------------------------------------------
    '''

    text_file = open(os.path.join(final_dir, "Readme_Images.txt"), "w")
    text_file.write(content)
    text_file.close()

if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(e)
