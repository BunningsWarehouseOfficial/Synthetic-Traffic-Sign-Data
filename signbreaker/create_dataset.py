def main():
    import os
    import sys
    import ntpath
    import shutil
    import argparse
    from datetime import datetime
    import random

    import cv2
    import json
    from collections import defaultdict
    from PIL import ImageFile
    from pathlib import Path
    import glob

    from bg_image import BgImage
    from damage import damage_image
    from utils import load_paths, load_files, scale_image, delete_background, to_png, dir_split
    from detect_light_src import find_light_source
    import manipulate
    import generate
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    parser = argparse.ArgumentParser(description='Create a synthetically generated traffic sign dataset')
    parser.add_argument('--output_dir', type=str, default=os.path.join(current_dir, 'SGTS_Dataset'))
    
    args = parser.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    os.chdir(current_dir)

    #TODO: Add an example download link for a dataset of backgrounds that have no real signs on them

    #TODO: Documentation in the dataset readme for what the tag of each damage type means

    # Open and validate config file
    import yaml
    with open("config.yaml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Input validation of config file
    valid_final = ['process', 'damage', 'transform', 'manipulate', 'dataset']
    valid_man = ['exposure', 'fade']
    valid_dmg = ['original', 'quadrant', 'big_hole', 'bullet_holes', 'graffiti', 'bend', 'tinted_yellow', 'grey']
    if config['sign_width'] <= 0:
        raise ValueError("Config error: 'sign_width' must be > 0.\n")
    if not config['final_op'] in valid_final:
            raise ValueError(f"Config error: '{config['final_op']}' is an invalid final_op value.\n")
    if config['num_transform'] < 0 or config['num_transform'] > 15:
        raise ValueError("Config error: must have 0 <= 'num_transform' <= 15.\n")
    if not config['man_method'] in valid_man:
        raise ValueError(f"Config error: 'man_method' must be either '{valid_man[0]}' or '{valid_man[1]}'.\n")
    for dmg in config['num_damages']:
        if not dmg in valid_dmg:
            raise ValueError(f"Config error: '{dmg}' is an invalid damage type.\n")

    b_params = config['bullet_holes']
    if b_params['min_holes'] <= 0 or b_params['max_holes'] <= 0:
        raise ValueError(f"Config error: 'bullet_holes' parameters must be > 0.\n")
    if b_params['min_holes'] > b_params['max_holes']:
        raise ValueError(f"Config error: 'bullet_holes:min_holes' must be <= 'bullet_holes:max_holes'.\n")
    if (b_params['target'] <= 0.0 or b_params['target'] >= 1.0) and b_params['target'] != -1:
        raise ValueError(f"Config error: 'bullet_holes:target' must be either -1 or in the range (0.0,1.0).\n")

    g_params = config['graffiti']
    for g_param in g_params:
        if g_params[g_param] <= 0.0 or g_params[g_param] > 1.0:
            raise ValueError(f"Config error: must have 0.0 < 'graffiti:{g_param}' <= 1.0.\n")
    if g_params['initial'] > g_params['final']:
        raise ValueError("Config error: 'graffiti:initial' must be <= 'graffiti:final'.\n")

    print("Generating dataset using the 'config.yaml' configuration.\n")

    # Directory names excluded from config.yaml to make use of .gitignore simpler
    base_dir        = "Sign_Templates"
    input_dir       = os.path.join(base_dir, "1_Input")
    processed_dir   = os.path.join(base_dir, "2_Processed")
    damaged_dir     = os.path.join(base_dir, "3_Damaged")
    transformed_dir = os.path.join(base_dir, "4_Transformed")
    manipulated_dir = os.path.join(base_dir, "5_Manipulated")
    bg_dir    = "Backgrounds"
    final_dir = args.output_dir  # SGTSD stands for "Synthetically Generated Traffic Sign Dataset"

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

    # Seed the random number generator
    random.seed(config['seed'])
    
    
    ##################################
    ###  Background Preprocessing  ###
    ##################################
    background_paths = glob.glob(f"{bg_dir}{os.sep}**{os.sep}*.png", recursive=True) + \
        glob.glob(f"{bg_dir}{os.sep}**{os.sep}*.jpg", recursive=True)
    background_images = []
    for ii, path in enumerate(background_paths):
        print(f"Processing Background Images: {float(ii) / float(len(background_paths)):06.2%}", end='\r')
        background_images.append(BgImage(path))
    print(f"Processing Background Images: 100.00%\r\n")

        
    #############################
    ###  IMAGE PREPROCESSING  ###
    #############################
    # Rescale images and make white backgrounds transparent
    paths = load_files(input_dir)
    for path in paths:
        _, filename = ntpath.split(path)
        name, extension = filename.rsplit('.', 1)
        
        img = scale_image(path, config['sign_width']) # Rescale the image

        # Remove the extension and save as a png
        
        save_path = os.path.join(processed_dir, name) + ".png"
        img.save(save_path)

        if not img.mode[-1] == 'A':
            delete_background(save_path, save_path) # Overwrite the newly rescaled image

    if config['final_op'] == 'process':
        return



    #########################
    ###  APPLYING DAMAGE  ###
    #########################
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
        damaged_data.append(damage_image(image_path, damaged_dir, config, background_images))
        ii += 1
    print(f"Damaging signs: 100.0%\r\n")
    damaged_data = [cell for row in damaged_data for cell in row]  # Flatten the list
    # else:
    #     print("Reusing pre-existing damaged signs")

    if config['final_op'] == 'damage':
        return



    #FIXME: 'transform_type' final label keeps having value None despite synth_image.set_transformation() working
    ##################################
    ###  APPLYING TRANSFORMATIONS  ###
    ##################################
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

    if config['final_op'] == 'transform':
        return



    ####################################
    ###  MANIPULATING EXPOSURE/FADE  ###
    ####################################
    if os.path.exists(manipulated_dir):
        shutil.rmtree(manipulated_dir)
    os.mkdir(manipulated_dir)

    ImageFile.LOAD_TRUNCATED_IMAGES = True  #TODO: Is this line needed?
    for bg_folders in load_paths(bg_dir):
        to_png(bg_folders)
        
    background_paths = glob.glob(f"{bg_dir}{os.sep}**{os.sep}*.png", recursive=True)
        
    #TODO: Can do checks for damage type in below functions to avoid funky results cancelling manipulation for just those types
    if config['man_method'] == 'exposure':
        manipulated_data = manipulate.exposure_manipulation(transformed_data, background_paths, manipulated_dir)
        manipulate.find_useful_signs(manipulated_data, manipulated_dir, damaged_dir) #TODO: MAKE SURE TO DO THIS, DELETING SYNTH_IMAGE OBJECTS ALONG THE WAY
    else:
        raise NotImplementedError('Only exposure method is currently implemented')
        # manipulated_data = manipulate.fade_manipulation(transformed_data, background_paths, manipulated_dir)

    # Delete SynthImage objects for any signs that were removed
    manipulated_data[:] = [x for x in manipulated_data if os.path.exists(x.fg_path)]
    
    # Prune dataset by randomly sampling from manipulated images
    if config['prune_dataset']['prune'] == 'true':
        max_images = config['prune_dataset']['max_images']
        images_dict = defaultdict(list)
        for img in manipulated_data:
            images_dict[os.path.dirname(img.fg_path)].append(img)
        images_dict = {k:random.sample(v, max_images) for k,v in images_dict.items() if len(v) > max_images}
        manipulated_data = [img for images_list in images_dict.values() for img in images_list]

    if config['final_op'] == 'manipulate':
        return



    ###############################
    ###  GENERATING FINAL DATA  ###
    ###############################
    images_dir = os.path.join(final_dir, "Images")
    labels_format = config['annotations']['type']
    damage_labelling = config['annotations']['damage_labelling'] == 'true'
    
    if labels_format == 'retinanet':
        labels_path = os.path.join(final_dir, "labels.txt")
        
    elif labels_format == 'coco':
        labels_path = os.path.join(final_dir, "_annotations.coco.json")
        classes = [int(Path(p).stem) for p in glob.glob(f'{processed_dir}{os.path.sep}*.png')]
        labels_dict = {'categories': [], 'images': [], 'annotations': []}
        labels_dict["categories"] += [{"id": 0, "name": "signs", "supercategory": "none"}]
        labels_dict["categories"] += [{'id:': c, 'name': str(c), 'supercategory': 'signs'} for c in sorted(classes)]
    about_path = os.path.join(final_dir, "generated_images_about.txt")

    # Clean and recreate the parent images directory
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    os.makedirs(images_dir)
        
    total_gen = len(manipulated_data)
    print(f"Files to be generated: {total_gen}")

    ii = 0
    labels_file = open(labels_path, "w")
    
    for synth_image in manipulated_data:
        print(f"Generating files: {float(ii) / float(total_gen):06.2%}", end='\r')
        
        c_num = synth_image.class_num
        d_type = synth_image.damage_type
        class_dir = os.path.join(images_dir, f"{c_num}", f"{c_num}_{d_type}")
        # Create the directory for each class+damage combination if it doesn't already exist
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        fg_path =  os.path.join(class_dir, f"{c_num}_{d_type}_{ii}")
        final_fg_path = fg_path + ".jpg"

        image = generate.new_data(synth_image)
        if labels_format == 'retinanet':
            synth_image.write_label_retinanet(labels_file, damage_labelling)
        elif labels_format == 'coco':
            synth_image.write_label_coco(labels_dict, ii, 
                        os.path.relpath(final_fg_path, final_dir), image.shape, damage_labelling)
        cv2.imwrite(final_fg_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        ii += 1
    print(f"Generating files: 100.0%\r\n")
    
    if labels_format == "coco":
        labels_dict['images'] = sorted(labels_dict['images'], key=lambda x: x['id'])
        labels_dict['annotations'] = sorted(labels_dict['annotations'], key=lambda x: x['id'])
        json.dump(labels_dict, labels_file, indent=4)
    labels_file.close()

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



    ###############################
    ###  CREATE DATASET README  ###
    ###############################
    content = '''
    -----------------------------------------------
    |                     -*-                     |
    |Synthetically Generated Traffic Sign Dataset |
    |                     -*-                     |
    -----------------------------------------------

    This directory contains the generated training
    set used for training a Convolutional Neural
    Network (CNN) to detect traffic signs.

    It may be used for any detector desired and it
    is not limited to a specific set of traffic
    sign templates.
    

    ----------------------------------------------
    Content
    ----------------------------------------------

    The number of examples is based on the number:
    - of traffic signs that were used as templates
    - of damage types applied to images
    - of transformations in image manipulations
    - of the brightness variation values used
    - of blending procedures
    - of background images used


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
    Label tags for each damage type
    ----------------------------------------------

    original:
        'no_damage=-1', meaning no damage.

    quadrant:
        'quadrant=XXX', where (X) is the missing
        quadrant with range [1,4].

    big_hole:
        'big_hole=AAA_BBB_CCC_DDD, where (A) is
        the angle of placement, (B) is the radius
        of the hole, and the x and y coordinates
        of the hole's centre are (C) and (D).

    bullet_holes:
        'bullet_holes=NNN', where (N) is the
        number of holes made in the sign.

    graffiti:
        'graffiti=TTT', where (T) is the *target*
        damage. As can be seen in the labels, the
        actual damage may be higher or lower.

    bend:
        'bend=LLL_RRR', where (L) and (R) are the
        angle of bending on the left and right
        sides of the sign respectively.


    ----------------------------------------------
    Additional information
    ----------------------------------------------

    Contact Emails:
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
