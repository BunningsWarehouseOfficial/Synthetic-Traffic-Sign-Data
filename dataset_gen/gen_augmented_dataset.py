import argparse
import os
import json
import shutil
import argparse
import random
from datetime import datetime
from glob import glob

from utils import initialise_coco_anns

parser = argparse.ArgumentParser()

parser.add_argument('--orig_dataset', default=None, help='Path to dataset to be augmented')
parser.add_argument('--orig_annotations', default=None, help='Annotations for original dataset')
parser.add_argument('--augment_dataset', default=None, help='Path to dataset that will be used to augment the original dataset')
parser.add_argument('--augment_annotations', default=None, help='Annotations for augment dataset')
parser.add_argument('--datasets_dir', default=None, help='Path to directory storing the dataset')
parser.add_argument('--augmentation', default=0.25, help='Proportion of the original dataset to be augmented')
parser.add_argument('--seed', default=0, help='Seed for random.sample function')


def extend_annotations(final_annotations, image_paths, annotations):
    image_paths = set([os.path.basename(p) for p in image_paths])
    image_id = len(final_annotations["images"])
    annotation_id = len(final_annotations["annotations"])
    
    for img_json in annotations['images']:
        path = os.path.basename(img_json["file_name"])
        
        if path in image_paths:
            image_anns = filter(lambda ann: ann['image_id'] == img_json['id'], annotations['annotations'])
            img_json['id'] = image_id
            img_json['file_name'] = path
            final_anns['images'].append(img_json)
            
            for ann in image_anns:
                ann['id'] = annotation_id
                ann['image_id'] = image_id
                final_annotations['annotations'].append(ann)
                annotation_id += 1
            image_id += 1


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    aug_factor = args.augmentation
    orig_dataset = args.orig_dataset.rstrip('/')
    augment_dataset = args.augment_dataset.rstrip('/')
    
    augmentation_dir = os.path.join(args.datasets_dir, 'augmented')
    if not os.path.exists(augmentation_dir):
        os.mkdir(augmentation_dir)
    
    outdir = os.path.join(augmentation_dir, f'{aug_factor}_augmented_gtsdb')
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    
    with open(args.orig_annotations, 'r') as f:
        orig_anns = json.load(f)
    with open(args.augment_annotations, 'r') as f:
        augment_anns = json.load(f)
    classes = [cat['name'] for cat in orig_anns['categories'][1:]]
    final_anns = initialise_coco_anns(classes)
    
    orig_paths = glob(args.orig_dataset + '/**/*.jpg', recursive=True)
    augment_paths = glob(args.augment_dataset + '/**/*.jpg', recursive=True)
    num_train = len(orig_paths)
    
    orig_paths = random.sample(orig_paths, int((1 - aug_factor) * num_train))
    augment_paths = random.sample(augment_paths, int(num_train * aug_factor))
    final_paths = orig_paths + augment_paths
    random.shuffle(final_paths)
    
    extend_annotations(final_anns, orig_paths, orig_anns)
    extend_annotations(final_anns, augment_paths, augment_anns)
    
    outpath = os.path.join(outdir, '_annotations.coco.json')
    with open(outpath, 'w') as f:
        json.dump(final_anns, f, indent=4)
        
    for i, p in enumerate(final_paths):
        print(f"Copying files: {float(i) / float(len(final_paths)):06.2%}", end='\r')
        shutil.copyfile(p, os.path.join(outdir, os.path.basename(p)))
    
    