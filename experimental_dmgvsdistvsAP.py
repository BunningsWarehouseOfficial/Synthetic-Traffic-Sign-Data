import os
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from collections import Counter, defaultdict
from evaluation_metrics.detection_eval import BoundingBox, get_pascal_voc_metrics
from damage_experiment import get_metrics

current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--gt_file', default='/home/allenator/Pawsey-Internship/datasets/sgts_sequences/_single_annotations_array.npy', 
                    help='Ground truth annotations for dataset as a numpy file')
parser.add_argument('--eval_file', default='/home/allenator/Pawsey-Internship/eval_dir/sgts_sequences/efficientdet-d0.npy', 
                    help='File containing evaluated detections as a numpy file')
parser.add_argument('--num_frames', default=8, type=int, help='Number of frames per sequence in dataset')
parser.add_argument('--experiment', default='dist_damage', choices=['damage', 'distance', 'dist_damage'] , help='Type of experiment to evaluate')


def arr_split(gt_array, arr_to_split):
    gt_array = gt_array[np.argsort(gt_array[:, 0])] # Sort gt_array by image_id just in case
    split_dict = defaultdict(list)
    distances = set([])
    damages = set([])
    
    for i in range(len(arr_to_split)):
        image_id = arr_to_split[i, 0]
        dist = gt_array[int(image_id), 6]
        damage = round(gt_array[int(image_id), 5], 1)
        distances.add(dist); damages.add(damage)
        split_dict[(damage, dist)].append(arr_to_split[i])
            
    split_dict = {dist:np.array(arr) for dist, arr in split_dict.items()}
    return split_dict



def get_meshgrid(gt_array):
    damages = np.unique(np.around(gt_array[:, 5], 1))
    distances = np.unique(gt_array[:, 6])
    return damages, distances


def damage_distance_experiment(gt_array, pred_array):
    distances_gt = arr_split(gt_array, gt_array)
    distances_pred = arr_split(gt_array, pred_array)
    metrics_array = np.zeros((len(distances_gt), 9), dtype=np.float32)
    damages, distances = get_meshgrid(gt_array)
    surface_z = np.zeros((len(distances), len(damages)))
    
    for i, key in enumerate(distances_gt):
        dmg, dist = key[0], key[1]
        
        # gt detections array in format [image_id, xtl, ytl, width, height, damage, distance, class_id]
        detections_gt = [BoundingBox(image_name=det[0], label=det[-1], xtl=det[1], ytl=det[2], xbr=det[1] + det[3], ybr=det[2] + det[4]) 
                        for det in distances_gt[key]]
        # pred detections array in format [image_id, xtl, ytl, width, height, score, class_id]
        detections_pred = [BoundingBox(image_name=det[0], label=det[-1], xtl=det[1], ytl=det[2], xbr=det[1] + det[3], ybr=det[2] + det[4], score=det[5])
                        for det in distances_pred[key]]
        
        metrics_array[i, 0:2] = [dmg, dist]
        metrics_array[i, 2:] = get_metrics(detections_gt, detections_pred)
        x = np.where(np.isclose(damages, dmg))[0]; y = np.where(np.isclose(distances, dist))[0]
        surface_z[y, x] = metrics_array[i, 3]
        
    columns = ['Damage', 'Distance', 'AP40', 'AP50', 'AP60', 'AP70', 'AP80', 'AP90', 'mAP']
    df = pd.DataFrame(metrics_array, columns=columns)
    df.sort_values(by=['Damage', 'Distance'], inplace=True, ascending=True)
    return df, (damages, distances, surface_z)


if __name__ == '__main__':
    args = parser.parse_args()
    gt_array = np.array(np.load(args.gt_file), dtype=np.float32)
    pred_array = np.array(np.load(args.eval_file), dtype=np.float32)
    
    df, mesh = damage_distance_experiment(gt_array, pred_array)
    print(df)
    x, y, z = mesh
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])
    fig.show()