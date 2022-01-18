import os
import json
import numpy as np
from datetime import datetime


def initialise_coco_anns(classes):
    labels = {}
    labels["info"] = {
        "year": "2021",
        "version": "1",
        "description": "Dataset of synthetically generated damaged traffic signs",
        "contributor": "Curtin University",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
    }
    labels["licenses"] = [{
        "id": 1,
        "url": "https://opensource.org/licenses/MIT",
        "name": "MIT License"
    }]
    labels["categories"] = [
        {
            "id": 0,
            "name": "signs",
            "supercategory": "none"
        }
    ]
    for i, c in enumerate(classes):
        labels["categories"].append(
            {
                "id": i + 1,
                "name": c, 
                "supercategory": "signs"
            }
        )
    labels["images"] = []
    labels["annotations"] = []
    return labels


def convert_to_single_label(dataset_path, original_annotations, new_annotations, use_damages=False):
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
        if use_damages:  
            annotations_array = np.array([[a["id"], a["image_id"], a["bg_id"], a["bbox"][0], a["bbox"][1], a["bbox"][2], a["bbox"][3], a["damage"], a["category_id"]] 
                                        for a in a_json['annotations']])
        else:
            annotations_array = np.array([[a["id"], a["image_id"], a["bbox"][0], a["bbox"][1], a["bbox"][2], a["bbox"][3], a["category_id"]] 
                                        for a in a_json['annotations']])
            
        with open(os.path.join(dataset_path, '_single_annotations_array.npy'), 'wb') as f:
            np.save(f, annotations_array)



