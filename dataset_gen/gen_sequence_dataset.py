import argparse
import os
import sys
import glob
import json
import shutil
from tqdm import tqdm
from pathlib import Path

from sequence_gen.create_sequences_auto import produce_anchors, SIGN_COORDS
from utils import initialise_coco_anns, convert_to_single_label, write_label_coco

parser = argparse.ArgumentParser()

parser.add_argument("bg_dir", type=str, help="path to background image directory")
parser.add_argument("damaged_sign_dir", type=str, help="path to damaged sign directory")
parser.add_argument("original_sign_dir", type=str, help="path to directory of original signs")
parser.add_argument("--max_dist", type=int, help="endpoint distance of sequence", default=20)
parser.add_argument("-n", "--num_frames", type=int, help="number of frames generated for each sequence", default=8)
parser.add_argument("-o", "--out_dir", type=str, help="path to output directory of sequences", default='./SGTS_sequences')

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

from signbreaker.utils import *


def create_sequence(bg_img, fg_img, bg_name, fg_name, sequence_id):
    anchors = produce_anchors(bg_img.shape, SIGN_COORDS['size'], SIGN_COORDS['x'], SIGN_COORDS['y'], 
                                4, args.max_dist, args.num_frames)
    bounding_boxes = []
    image_paths = []
    
    for frame, anchor in enumerate(anchors):
        save_path = f"{args.out_dir}/{sequence_id + frame}-{bg_name}-{fg_name}-{frame}.jpg"
        scaled_fg_img = cv2.resize(fg_img, (anchor.size, anchor.size))
        new_img = overlay(scaled_fg_img, bg_img, anchor.screen_x, anchor.screen_y)
        cv2.imwrite(save_path, new_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        bounding_boxes.append([anchor.screen_x, anchor.screen_y, anchor.size, anchor.size])
        image_paths.append(save_path)
    return (image_paths, bounding_boxes)


if __name__ == '__main__':
    args = parser.parse_args(['../signbreaker/Backgrounds/GTSDB', '../signbreaker/Sign_Templates/3_Damaged', '../signbreaker/Sign_Templates/2_Processed'])

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.mkdir(args.out_dir)
        
    annotations = initialise_coco_anns(os.listdir(args.original_sign_dir))
    sign_paths = glob.glob(args.damaged_sign_dir + '/*/*.png')
    bg_paths = glob.glob(args.bg_dir + "/*.png")
    image_id = 0
    
    for bg_id, bg_path in enumerate(bg_paths):
        bg_name = Path(bg_path).stem
        bg_img = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
        height, width, _ = bg_img.shape
        print(f"Generating sequences for background image {bg_name}")
        
        for sign_path in tqdm(sign_paths):
            sign_name = Path(sign_path).stem
            cls_name = sign_name.split('_')[0]
            original_sign_path = os.path.join(args.original_sign_dir, f'{cls_name}.png')
            
            sign_img = cv2.imread(sign_path, cv2.IMREAD_UNCHANGED)
            original_sign_img = cv2.imread(original_sign_path, cv2.IMREAD_UNCHANGED)
            image_paths, bounding_boxes = create_sequence(bg_img, sign_img, bg_name, sign_name, image_id)
            
            for i in range(len(image_paths)):
                # Format [x, y, width, height, distance]
                bbox = bounding_boxes[i]
                image_i = cv2.resize(sign_img, (bbox[2], bbox[3]))
                original_image_i = cv2.resize(original_sign_img, (bbox[2], bbox[3]))
                damage_i = calc_damage(original_image_i, image_i)
                
                file_path = os.path.basename(image_paths[i])
                write_label_coco(annotations, file_path, image_id, int(cls_name), bbox, damage_i, (height, width))
                image_id += 1
                
    annotations_path = os.path.join(args.out_dir, "_annotations.coco.json")
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=4)
    convert_to_single_label(args.out_dir, '_annotations.coco.json', '_single_annotations.coco.json', use_damages=True)
            
            
        
        