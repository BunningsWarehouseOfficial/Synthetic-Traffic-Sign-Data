import argparse
import os
import glob
import json
import shutil
from tqdm import tqdm
from datetime import datetime
from signbreaker.utils import *
from signbreaker.create_sequences_auto import create_sequence
from pathlib import Path

current_dir = os.path.dirname(os.path.realpath(__file__))
default_outdir = os.path.join(current_dir, "sgts_sequences")

parser = argparse.ArgumentParser()

parser.add_argument("bg_dir", type=str, help="path to background image directory")
parser.add_argument("damaged_sign_dir", type=str, help="path to damaged sign directory")
parser.add_argument("original_sign_dir", type=str, help="path to directory of original signs")
parser.add_argument("-n", "--num_frames", type=int, help="number of frames generated for each sequence", default=8)
parser.add_argument("-o", "--out_dir", type=str, help="path to output directory of sequences", default=default_outdir)


def initialise_annotations(num_classes):
    annotations = {}
    annotations["info"] = {
        "year": "2021",
        "version": "1",
        "description": "Dataset of sequences of synthetically generated damaged signs",
        "contributor": "Curtin University",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    }
    annotations["licenses"] = {
        "id": 1,
        "url": "https://opensource.org/licenses/MIT",
        "name": "MIT License"
    }
    annotations["categories"] = [
        {
            "id": 0,
            "name": "signs",
            "supercategory": "none"
        }
    ]
    for i in range(num_classes):
        annotations["categories"].append(
            {
                "id": i + 1,
                "name": "placeholder", #TODO: find a way to get the name of the sign
                "supercategory": "signs"
            }
        )
    annotations["gt"] = []
    return annotations


if __name__ == '__main__':
    args = parser.parse_args()
    
    OUTDIR = args.out_dir
    if os.path.exists(OUTDIR):
        shutil.rmtree(OUTDIR)
    os.mkdir(OUTDIR)
        
    annotations = initialise_annotations(len(os.listdir(args.original_sign_dir)))
    sign_paths = glob.glob(args.damaged_sign_dir + '/*/*.png')
    bg_paths = glob.glob(args.bg_dir + "/*.png")
    image_id = 0
    sequence_id = 0
    
    for bg_path in bg_paths:
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
            image_paths, bounding_boxes = create_sequence(bg_img, sign_img, bg_name, sign_name, image_id, OUTDIR, num_frames=args.num_frames)
            annotations["gt"].append({'images': [], 'annotations': [], 'damage': round(calc_damage(sign_img, original_sign_img), 1)})
            
            for i in range(len(image_paths)):
                annotations['gt'][sequence_id]["images"].append(
                    {
                        "id": image_id,
                        "width": width,
                        "height": height,
                        "file_name": os.path.basename(image_paths[i])
                    }
                )
                annotations['gt'][sequence_id]["annotations"].append(
                    {
                        "id": image_id,        # one annotation per image
                        "image_id": image_id,
                        "category_id": int(cls_name) + 1,
                        "bbox": bounding_boxes[i],
                        "iscrowd": 0,
                        "area": bounding_boxes[i][2] ** 2,
                        "segmentation": [],
                    }
                )
                image_id += 1
            sequence_id += 1
                
    annotations_path = os.path.join(OUTDIR, "_annotations.coco.json")
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=4)
            
            
        
        