import os
import argparse
import json
import numpy as np
import pandas as pd
import plotly.express as px

from collections import defaultdict
from experiments.detection_eval import BoundingBox, get_pascal_voc_metrics, Box
from tqdm import tqdm

from experiments.damage_experiment import damage_experiment, distance_experiment


current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--gt_file', default='/home/allenator/Pawsey-Internship/datasets/sgts_sequences/_single_annotations_array.npy', 
                    help='Ground truth annotations for dataset as a numpy file')
parser.add_argument('--eval_files', default='/home/allenator/Pawsey-Internship/eval_dir/sgts_sequences_8/augmented.json',
                    help='Json of augment_level:file_path pairs')
parser.add_argument('--num_frames', default=8, type=int, help='Number of frames per sequence the dataset')

if __name__ == "__main__":
    args = parser.parse_args()
    gt_array = np.array(np.load(args.gt_file), dtype=np.float32)
    
    with open(args.eval_files) as f:
        eval_json = json.load(f)
    
    augment_dict = {}
    
    for aug in tqdm(eval_json):
        npy_path = eval_json[aug]
        pred_array = np.array(np.load(npy_path), dtype=np.float32)
        augment_dict[aug] = \
            {
            'damage_experiment': damage_experiment(gt_array, pred_array),
            'distance_experiment': distance_experiment(gt_array, pred_array)
            }
    
            
