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
default_background_dir = os.path.join(current_dir, "signbreaker/Backgrounds/GTSDB")
default_damaged_dir = os.path.join(current_dir, "signbreaker/Sign_Templates/3_Damaged")
default_original_sign_dir = os.path.join(current_dir, "signbreaker/Sign_Templates/2_Processed/")

parser = argparse.ArgumentParser()

parser.add_argument("bg_dir", type=str, help="path to background image directory")
parser.add_argument("damaged_sign_dir", type=str, help="path to damaged sign directory")
parser.add_argument("original_sign_dir", type=str, help="path to directory of original signs")
parser.add_argument("--max_dist", type=int, help="endpoint distance of sequence", default=20)
parser.add_argument("-n", "--num_frames", type=int, help="number of frames generated for each sequence", default=8)
parser.add_argument("-o", "--out_dir", type=str, help="path to output directory of sequences", default=default_outdir)


def initialise_annotations(num_classes):
    annotations = {}
    annotations["info"] = {
        "year": "2021",
        "version": "1",
        "description": "Dataset of sequences of synthetically generated damaged signs",
        "contributor": "Curtin University",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "frames_per_sequence": args.num_frames
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
    annotations["images"] = []
    annotations["annotations"] = []
    return annotations


def convert_to_single_label(dataset_path, original_annotations, new_annotations):
    with open(os.path.join(dataset_path, original_annotations), 'r') as a_file:
        a_json = json.load(a_file)
        
        a_json["categories"] = [
            {
                "id": 0,
                "name": "signs",
                "supercategory": "none"
            },
            {
                "id": 1,
                "name": "traffic_sign",
                "supercategory": "signs"
            }
        ]
        
        for annotation in a_json['annotations']:
            annotation['category_id'] = 1
        
        with open(os.path.join(dataset_path, new_annotations), 'w') as f:
            json.dump(a_json, f, indent=4)
        
        # Create a npy file to store ground truths, for more efficient evaluation    
        annotations_array = np.array([[a["id"], a["bg_id"], a["sign_id"], a["bbox"][0], a["bbox"][1], a["bbox"][2], a["bbox"][3], a["damage"], a["distance"], a["category_id"]] 
                                    for a in a_json['annotations']])
        
        with open(os.path.join(dataset_path, '_single_annotations_array.npy'), 'wb') as f:
            np.save(f, annotations_array)


if __name__ == '__main__':
    args = parser.parse_args()

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.mkdir(args.out_dir)
        
    annotations = initialise_annotations(len(os.listdir(args.original_sign_dir)))
    sign_paths = glob.glob(args.damaged_sign_dir + '/*/*.png')
    bg_paths = glob.glob(args.bg_dir + "/*.png")
    image_id = 0
    
    for bg_id, bg_path in enumerate(bg_paths):
        bg_name = Path(bg_path).stem
        bg_img = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
        height, width, _ = bg_img.shape
        print(f"Generating sequences for background image {bg_name}")
        
        sign_id = 0    
        for sign_path in tqdm(sign_paths):
            sign_name = Path(sign_path).stem
            cls_name = sign_name.split('_')[0]
            original_sign_path = os.path.join(args.original_sign_dir, f'{cls_name}.png')
            
            sign_img = cv2.imread(sign_path, cv2.IMREAD_UNCHANGED)
            original_sign_img = cv2.imread(original_sign_path, cv2.IMREAD_UNCHANGED)
            image_paths, bounding_boxes = create_sequence(bg_img, sign_img, bg_name, sign_name, image_id, 
                                                        args.out_dir, num_frames=args.num_frames, max_dist=args.max_dist)
            
            for i in range(len(image_paths)):
                # Format [x, y, width, height, distance]
                bbox = bounding_boxes[i]
                image_i = cv2.resize(sign_img, (bbox[2], bbox[3]))
                original_image_i = cv2.resize(original_sign_img, (bbox[2], bbox[3]))
                damage_i = calc_damage(original_image_i, image_i)
                
                annotations["images"].append(
                    {
                        "id": image_id,
                        "background_name": bg_name,
                        "width": width,
                        "height": height,
                        "file_name": os.path.basename(image_paths[i])
                    }
                )
                annotations["annotations"].append(
                    {
                        "id": image_id,        # one annotation per image
                        "bg_id": bg_id,
                        "sign_id": sign_id,
                        "category_id": int(cls_name) + 1,
                        "bbox": bbox,
                        "iscrowd": 0,
                        "area": bbox[2] ** 2,
                        "segmentation": [],
                        "damage": damage_i,
                        "distance": bbox[-1]
                    }
                )
                image_id += 1
            sign_id += 1
                
    annotations_path = os.path.join(args.out_dir, "_annotations.coco.json")
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=4)
    convert_to_single_label(args.out_dir, '_annotations.coco.json', '_single_annotations.coco.json')
            
            
        
        