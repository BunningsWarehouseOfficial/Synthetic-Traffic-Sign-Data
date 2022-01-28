import os
import argparse
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm

from damage_experiment import damage_experiment, distance_experiment, sequence_experiment

current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--gt_file', default='/home/allenator/Pawsey-Internship/datasets/sgts_sequences_8/_single_annotations_array.npy', 
                    help='Ground truth annotations for dataset as a numpy file')
parser.add_argument('--eval_files', default='/home/allenator/Pawsey-Internship/eval_dir/sgts_sequences_8/augments.json',
                    help='Json of augment_level:file_path pairs')
parser.add_argument('--num_frames', default=8, type=int, help='Number of frames per sequence the dataset')
parser.add_argument('--experiment', default='damage', choices=['damage', 'distance', 'sequence'] , help='Type of experiment to evaluate')
parser.add_argument('--metric', default='mAP', choices=['AP50','mAP', 'Mean IOU', 'Mean Score'] , help='Type of metric to evaluate')

if __name__ == "__main__":
    args = parser.parse_args()
    
    is_distance_experiment = args.experiment == 'distance'
    is_sequence_experiment = args.experiment == 'sequence'
    is_damage_experiment = args.experiment == 'damage'
    
    gt_array = np.array(np.load(args.gt_file), dtype=np.float32)
    
    with open(args.eval_files) as f:
        eval_json = json.load(f)
    
    augment_dict = {}
    
    # get data frames for each augment level
    for aug in tqdm(eval_json):
        npy_path = eval_json[aug]
        pred_array = np.array(np.load(npy_path), dtype=np.float32)
        if args.experiment == 'damage':
            df = damage_experiment(gt_array, pred_array)
        elif args.experiment == 'distance':
            df = distance_experiment(gt_array, pred_array)
        elif args.experiment == 'sequence':
            df = sequence_experiment(gt_array, pred_array)
        augment_dict[aug] = df
    
    fig = px.scatter(title=args.experiment.capitalize() + ' vs ' + args.metric)
    
    # Axis labels   
    fig.update_yaxes(title=args.metric)
    if is_damage_experiment or is_sequence_experiment:
        fig.update_xaxes(title_text='Damage Ratio')
    elif is_distance_experiment:
        fig.update_xaxes(title_text='Area of Sign in Pixels')
            
    # plot given experiment
    for aug in augment_dict:
        df = augment_dict[aug]
        if is_damage_experiment or is_sequence_experiment:
            fig = fig.add_trace(go.Scatter(x = df['Damage'], y = df[args.metric], 
                                        name=aug, mode='lines+markers'))
        elif is_distance_experiment:
            fig = fig.add_trace(go.Scatter(x = df['Area'], y = df[args.metric], 
                                        name=aug, mode='lines+markers'))
    fig.show()
    
            
