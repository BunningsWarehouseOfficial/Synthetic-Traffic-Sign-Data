import os
import argparse
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from collections import Counter, defaultdict
from evaluation_metrics.detection_eval import BoundingBox, get_pascal_voc_metrics


current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--gt_file', default='/home/allenator/Pawsey-Internship/datasets/sgts_sequences/_single_annotations_array.npy', 
                    help='Ground truth annotations for dataset as a numpy file')
parser.add_argument('--eval_file', default='/home/allenator/Pawsey-Internship/eval_dir/sgts_sequences/efficientdet-d0.npy', 
                    help='File containing evaluated detections as a numpy file')
parser.add_argument('--num_frames', default=8, type=int, help='Number of frames per sequence in dataset')
parser.add_argument('--experiment', default='dist_damage', choices=['damage', 'distance', 'dist_damage'] , help='Type of experiment to evaluate')


def split_by_sequence(arr, num_frames):
    """
    Splits an array of detections into sequences of detections.
    """
    split_indices = []
    # Sort array by image_id
    arr = arr[np.argsort(arr[:, 0])]
    image_ids = arr[:, 0]
    for i in range(len(image_ids) - 1):
        if image_ids[i] % num_frames == 0 and image_ids[i] != image_ids[i + 1] and int(image_ids[i]) != 0:
            split_indices.append(i)
    split_arrs = np.split(arr.copy(), split_indices)
    return split_arrs


def map_seq_to_dmg(sequences_arr):
    sequence_damages = [0] * len(sequences_arr)
    damage_level_count = Counter()
    
    for i in range(len(sequences_arr)):
        detections = sequences_arr[i]
        damage_level = round(np.mean(detections[:, 5]), 1)
        damage_level_count[damage_level] += 1
        sequence_damages[i] = damage_level
    return sequence_damages, damage_level_count


def split_by_distance(gt_array, arr_to_split, also_split_by_damage=False, get_meshgrid_shape=False):
    gt_array = gt_array[np.argsort(gt_array[:, 0])] # Sort gt_array by image_id just in case
    split_dict = defaultdict(list)
    distances = set([])
    damages = set([])
    
    for i in range(len(arr_to_split)):
        image_id = arr_to_split[i, 0]
        dist = gt_array[int(image_id), 6]
        damage = round(gt_array[int(image_id), 5], 1)
        distances.add(dist); damages.add(damage)
        
        if also_split_by_damage:
            split_dict[(damage, dist)].append(arr_to_split[i])
        else:
            split_dict[dist].append(arr_to_split[i])
            
    split_dict = {dist:np.array(arr) for dist, arr in split_dict.items()}
    if get_meshgrid_shape:
        return split_dict, len(distances), (damages)
    else:
        return split_dict


def get_metrics(gt, pred):
    """
    Calculates the metrics for a given set of ground truth and predicted detections.
    Metrics in the format [AP50, AP75, AP95, max precision, max recall, min precision, min recall]
    """    
    metrics = np.zeros(7)
    metrics[0] = get_pascal_voc_metrics(gt, pred, iou_threshold=0.40).ap
    metrics[1] = get_pascal_voc_metrics(gt, pred, iou_threshold=0.50).ap
    metrics[2] = get_pascal_voc_metrics(gt, pred, iou_threshold=0.60).ap
    metrics[3] = get_pascal_voc_metrics(gt, pred, iou_threshold=0.70).ap
    metrics[4] = get_pascal_voc_metrics(gt, pred, iou_threshold=0.80).ap
    metrics[5] = get_pascal_voc_metrics(gt, pred, iou_threshold=0.90).ap
    metrics[6] = np.mean(metrics[:7])
    return metrics


def prune_detections(detections_array, max_detections=50):
    sorted_indices = np.argsort(detections_array[:, 5], axis=0)
    pruned_arr = detections_array[sorted_indices][::-1]
    return pruned_arr[:max_detections]


def damage_experiment(gt_array, pred_array):
    sequences_gt = split_by_sequence(gt_array, args.num_frames)
    sequences_pred = split_by_sequence(pred_array, args.num_frames)
    
    metrics_array = np.zeros((11, 8), dtype=np.float32)
    metrics_array[:, 0] = np.arange(0, 110, 10)
    
    # Computes average damage level for each sequence and incerements count for each damage 
    # level over all sequences
    sequence_damages, damage_level_count = map_seq_to_dmg(sequences_gt)
    
    # iterate over each image sequence
    for i in range(len(sequences_gt)):
        damage_level = sequence_damages[i]
        
        # gt detections array in format [image_id, xtl, ytl, width, height, damage, distance, class_id]
        detections_gt = [BoundingBox(image_name=det[0], label=det[-1], xtl=det[1], ytl=det[2], xbr=det[1] + det[3], ybr=det[2] + det[4]) 
                        for det in sequences_gt[i]]
        # pred detections array in format [image_id, xtl, ytl, width, height, score, class_id]
        detections_pred = [BoundingBox(image_name=det[0], label=det[-1], xtl=det[1], ytl=det[2], xbr=det[1] + det[3], ybr=det[2] + det[4], score=det[5])
                        for det in sequences_pred[i]]
        
        metrics = get_metrics(detections_gt, detections_pred)
        # Converting from 0-1 damage range to 0-10 index range
        metrics_array[int(damage_level * 10), 1:] += metrics / damage_level_count[damage_level]
    
    columns = ['Damage', 'AP40', 'AP50', 'AP60', 'AP70', 'AP80', 'AP90', 'mAP']
    
    dataless_rows = (metrics_array[:, 6] == 0.0).nonzero()[0]
    metrics_array = np.delete(metrics_array, dataless_rows, axis=0)
    return pd.DataFrame(metrics_array, columns=columns)


def distance_experiment(gt_array, pred_array):
    distances_gt = split_by_distance(gt_array, gt_array)
    distances_pred = split_by_distance(gt_array, pred_array)
    metrics_array = np.zeros((len(distances_gt), 8), dtype=np.float32)
    
    for i, dist in enumerate(distances_gt):
        # gt detections array in format [image_id, xtl, ytl, width, height, damage, distance, class_id]
        detections_gt = [BoundingBox(image_name=det[0], label=det[-1], xtl=det[1], ytl=det[2], xbr=det[1] + det[3], ybr=det[2] + det[4]) 
                        for det in distances_gt[dist]]
        # pred detections array in format [image_id, xtl, ytl, width, height, score, class_id]
        detections_pred = [BoundingBox(image_name=det[0], label=det[-1], xtl=det[1], ytl=det[2], xbr=det[1] + det[3], ybr=det[2] + det[4], score=det[5])
                        for det in distances_pred[dist]]
        metrics_array[i, 0] = dist
        metrics_array[i, 1:] = get_metrics(detections_gt, detections_pred)
        
    columns = ['Distance', 'AP40', 'AP50', 'AP60', 'AP70', 'AP80', 'AP90', 'mAP']
    return pd.DataFrame(metrics_array, columns=columns)


def get_meshgrid(gt_array):
    damages = np.unique(np.around(gt_array[:, 5], 1))
    distances = np.unique(gt_array[:, 6])
    return damages, distances
        

def damage_distance_experiment(gt_array, pred_array):
    distances_gt = split_by_distance(gt_array, gt_array, also_split_by_damage=True)
    distances_pred = split_by_distance(gt_array, pred_array, also_split_by_damage=True)
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
    
    if args.experiment == 'damage':
        df = damage_experiment(gt_array, pred_array)
        print(df)
        fig = px.line(df, x='Damage', y='AP70', title='Average Precision (AP) vs. Damage Level')
        
    elif args.experiment == 'distance':
        df = distance_experiment(gt_array, pred_array)
        print(df)
        fig = px.line(df, x='Distance', y='AP50', title='Average Precision (AP) vs. Distance from camera')
        
    else:
        df, mesh = damage_distance_experiment(gt_array, pred_array)
        print(df)
        x, y, z = mesh
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])
    fig.show()
        
        
    
        
        
    
    
    
    
    
