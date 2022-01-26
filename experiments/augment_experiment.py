import os
import argparse
import json
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from damage_experiment import damage_experiment, distance_experiment

current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--gt_file', default='/home/allenator/Pawsey-Internship/datasets/sgts_sequences_8/_single_annotations_array.npy', 
                    help='Ground truth annotations for dataset as a numpy file')
parser.add_argument('--eval_files', default='/home/allenator/Pawsey-Internship/eval_dir/sgts_sequences_8/augments.json',
                    help='Json of augment_level:file_path pairs')
parser.add_argument('--num_frames', default=8, type=int, help='Number of frames per sequence the dataset')
parser.add_argument('--experiment', default='damage', choices=['damage', 'distance'] , help='Type of experiment to evaluate')

if __name__ == "__main__":
    args = parser.parse_args()
    gt_array = np.array(np.load(args.gt_file), dtype=np.float32)
    
    with open(args.eval_files) as f:
        eval_json = json.load(f)
    
    augment_dict = {}
    
    # get data frames
    for aug in tqdm(eval_json):
        npy_path = eval_json[aug]
        pred_array = np.array(np.load(npy_path), dtype=np.float32)
        augment_dict[aug] = \
            {
            'damage': damage_experiment(gt_array, pred_array, args.num_frames),
            'distance': distance_experiment(gt_array, pred_array)
            }
            
    # plot given experiment
    fig = go.Figure()
    for aug in augment_dict:
        df = augment_dict[aug][args.experiment]
        if args.experiment == 'damage':
            fig = fig.add_trace(go.Scatter(x = df['Damage'], y = df['mAP'], 
                                        name=aug, mode='lines+markers'))
        elif args.experiment == 'distance':
            fig = fig.add_trace(go.Scatter(x = df['Sign Width'], y = df['mAP'], 
                                        name=aug, mode='lines+markers'))
    fig.show()
    
            
