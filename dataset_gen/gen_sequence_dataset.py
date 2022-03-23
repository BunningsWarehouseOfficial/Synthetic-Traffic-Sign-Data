"""Module to generate a dataset consisting of sequences. Each sequence is a set of images where a foreground (sign) is
interpolated across a background to mimic changing distance.
"""

import argparse
from ast import parse
import os
import sys
import glob
import json
import shutil
import cv2
import random
from tqdm import tqdm
from pathlib import Path
from sequence_gen.create_sequences_auto import produce_anchors, get_world_coords
from utils import initialise_coco_anns, convert_to_single_label, write_label_coco

parser = argparse.ArgumentParser()

parser.add_argument("--min_dist", type=int, help="startpoint distance of sequence", default=4)
parser.add_argument("--max_dist", type=int, help="endpoint distance of sequence", default=20)
parser.add_argument("--max_fg_height", type=int, default=0.15, help="maximum sign height proportional to window height")
parser.add_argument("-n", "--num_frames", type=int, help="number of frames generated for each sequence", default=8)
parser.add_argument("-o", "--out_dir", type=str, help="path to output directory of sequences", default='./SGTS_Sequences')
parser.add_argument("--augment", type=str, help="augmentation type", choices=['manipulated', 'transformed', 'none'], default='transformed')

args = parser.parse_args()
args.out_dir = os.path.abspath(args.out_dir)

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

from signbreaker.utils import *

os.chdir(current_dir)
bg_dir = '../signbreaker/Backgrounds/GTSDB'
manipulated_sign_dir = '../signbreaker/Sign_Templates/5_Manipulated/GTSDB'
transformed_sign_dir = '../signbreaker/Sign_Templates/4_Transformed'
damaged_sign_dir     = '../signbreaker/Sign_Templates/3_Damaged'
original_sign_dir    = '../signbreaker/Sign_Templates/2_Processed'

# TODO: Having fixed coordinates is an issue, especially for light sourced bent signs which are assigned a specific
#       coordinate that is ignored here
COORDS = {'x':0.75, 'y':0.45}  # Default proportional (0-1) sign coordinates in 2D plane


def create_sequence(bg_img, fg_img, bg_name, fg_name, sequence_id):
    # Randomly place initial sign in bottom third
    
    fg_img = remove_padding(fg_img)
    bg_height, bg_width, _ = bg_img.shape
    fg_height, fg_width, _ = fg_img.shape

    sign_aspect = fg_width / fg_height
    y_size = args.max_fg_height
    x_size = sign_aspect * args.max_fg_height
    
    x_world, y_world, x_wsize, y_wsize = get_world_coords(bg_width / bg_height, 
                    COORDS['x'], COORDS['y'], args.min_dist, (x_size, y_size))
    
    anchors = produce_anchors(bg_img.shape, x_world, y_world, (x_wsize, y_wsize),
                                args.min_dist, args.max_dist, args.num_frames)
    bounding_boxes = []
    image_paths = []
    
    for frame, anchor in enumerate(anchors):
        save_path = os.path.join(args.out_dir, f"{sequence_id + frame}-{bg_name}-{fg_name}-{frame}.jpg")
        scaled_fg_img = cv2.resize(fg_img, (anchor.width, anchor.height), interpolation=cv2.INTER_AREA)
        new_img = overlay(scaled_fg_img, bg_img, anchor.screen_x, anchor.screen_y)
        cv2.imwrite(save_path, new_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        bounding_boxes.append([anchor.screen_x, anchor.screen_y, anchor.width, anchor.height])
        image_paths.append(save_path)
    return (image_paths, bounding_boxes)


def get_damage(damaged_img, original_img, bbox):
    import yaml
    with open('../signbreaker/config.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    image_i = cv2.resize(damaged_img, (bbox[2], bbox[3]))
    original_image_i = cv2.resize(original_img, (bbox[2], bbox[3]))
    return calc_damage(image_i, original_image_i, config['damage_measure_method'])


def generate_augmented(image_id, bg_img, bg_name, bg_dims, annotations, augment='transformed'):
    if augment == 'manipulated':
        sign_dirs = glob.glob(os.path.join(manipulated_sign_dir, f'BG_{bg_name}') + '/*/*/')
    elif augment == 'transformed':
        sign_dirs = glob.glob(transformed_sign_dir + '/*/*/')
    sign_dirs = sorted(sign_dirs, key=lambda p: int(Path(p).stem.split('_')[0]))

    for sign_dir in tqdm(sign_dirs):
        if glob.glob(sign_dir + '/*.png') != []:
            sign_path = random.choice(glob.glob(sign_dir + '/*.png'))
        else:
            continue
        
        sign_name = Path(sign_path).parts[-2]
        cls_name = sign_name.split('_')[0]
        
        damaged_sign_path = os.path.join(damaged_sign_dir, cls_name, sign_name + '.png')
        original_sign_path = os.path.join(original_sign_dir, f'{cls_name}.png')
        
        sign_img = cv2.imread(sign_path, cv2.IMREAD_UNCHANGED)
        damaged_sign_img = cv2.imread(damaged_sign_path, cv2.IMREAD_UNCHANGED)
        original_sign_img = cv2.imread(original_sign_path, cv2.IMREAD_UNCHANGED)
        image_paths, bounding_boxes = create_sequence(bg_img, sign_img, bg_name, sign_name, image_id)
        
        for i in range(len(image_paths)):
            # Format [x, y, width, height, distance]
            damage_i = get_damage(damaged_sign_img, original_sign_img, bounding_boxes[i])
            file_path = os.path.basename(image_paths[i])
            write_label_coco(annotations, file_path, image_id, int(cls_name), bounding_boxes[i], damage_i, bg_dims)
            image_id += 1
    return image_id


def generate_damaged(image_id, bg_img, bg_name, bg_dims, annotations):
    sign_paths = glob.glob(damaged_sign_dir + '/*/*.png')

    for sign_path in tqdm(sign_paths):
        sign_name = Path(sign_path).stem
        cls_name = sign_name.split('_')[0]
        original_sign_path = os.path.join(original_sign_dir, f'{cls_name}.png')
        
        sign_img = cv2.imread(sign_path, cv2.IMREAD_UNCHANGED)
        original_sign_img = cv2.imread(original_sign_path, cv2.IMREAD_UNCHANGED)
        image_paths, bounding_boxes = create_sequence(bg_img, sign_img, bg_name, sign_name, image_id)
        
        for i in range(len(image_paths)):
            # Format [x, y, width, height, distance]
            damage_i = get_damage(sign_img, original_sign_img, bounding_boxes[i])
            file_path = os.path.basename(image_paths[i])
            write_label_coco(annotations, file_path, image_id, int(cls_name), bounding_boxes[i], damage_i, bg_dims)
            image_id += 1
    return image_id


if __name__ == '__main__':
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.mkdir(args.out_dir)
    
    classes = sorted([Path(p).stem for p in os.listdir(original_sign_dir)], key=lambda p: int(p))  
    annotations = initialise_coco_anns(classes)
    bg_paths = glob.glob(bg_dir + "/*.png")
    image_id = 0
    
    for bg_id, bg_path in enumerate(bg_paths):
        bg_name = Path(bg_path).stem
        bg_img = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
        height, width, _ = bg_img.shape
        print(f"Generating sequences for background image {bg_name}")
        if args.augment == 'none':
            image_id = generate_damaged(image_id, bg_img, bg_name, (height, width), annotations)
        else:
            image_id = generate_augmented(image_id, bg_img, bg_name, (height, width), annotations, args.augment)
                
    annotations_path = os.path.join(args.out_dir, "_annotations.coco.json")
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=4)
    convert_to_single_label(args.out_dir, '_annotations.coco.json', '_single_annotations.coco.json', use_damages=True)
            
            
        
        