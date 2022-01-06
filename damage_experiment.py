'''    
    This code evaluates given against the ground truth for the sgts_sequences dataset,
    evaluating each sequence seperately via various AP thresholds, averaging the results of sequences with the same
    (rounded to 1 decimal place) damage metric, and then producing a dataframe in the following format:
    _____________________________________________________________________________________
    |   Damage  |   AP40  |   AP50  |   AP60  |   AP70  |   AP80  |   AP90  |   APMean  |
    |___________|_________|_________|_________|_________|_________|_________|___________|
    |   0       |
    |   10      |                                                       
    |   20      |                                                                       
    |   30      |                                                   
    |   40      |      
    |   50      |     
    |   60      |      
    |   70      |      
    |   80      |     
    |   90      |  
        
    Author: Allen Antony                                                                     
'''

import os
import argparse
from posixpath import split
import numpy as np
import pandas as pd
import plotly.express as px

from collections import Counter
from evaluation_metrics.detection_eval import BoundingBox, get_pascal_voc_metrics


current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--gt_file', default='/home/allenator/Pawsey-Internship/datasets/sgts_sequences/_single_annotations_array.npy', 
                    help='Ground truth annotations for dataset as a numpy file')
parser.add_argument('--eval_file', default='/home/allenator/Pawsey-Internship/eval_dir/sgts_sequences/efficientdet-d0.npy', 
                    help='File containing evaluated detections as a numpy file')
parser.add_argument('--num_frames', default=8, type=int, help='Number of frames per sequence in dataset')


def split_array_by_sequence(arr, num_frames):
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


if __name__ == '__main__':
    args = parser.parse_args()
    sequences_gt = split_array_by_sequence(np.load(args.gt_file), args.num_frames)
    sequences_pred = split_array_by_sequence(np.load(args.eval_file), args.num_frames)
    sequences_pred = [prune_detections(detections) for detections in sequences_pred]
    
    sequence_damages = [0] * len(sequences_gt)
    metrics_array = np.zeros((11, 8), dtype=np.float32)
    metrics_array[:, 0] = np.arange(0, 110, 10)
    damage_level_count = Counter()
    
    for i in range(len(sequences_gt)):
        detections_gt = sequences_gt[i]
        damage_level = round(np.mean(detections_gt[:, 5]), 1)
        damage_level_count[damage_level] += 1
        sequence_damages[i] = damage_level
    
    # iterate over each image sequence
    for i in range(len(sequences_gt)):
        damage_level = sequence_damages[i]
        
        # gt detections array in format [image_id, xtl, ytl, width, height, damage, class_id]
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
    df = pd.DataFrame(metrics_array, columns=columns)
    
    # Draw plotly chart for damage vs AP
    fig = px.line(df, x='Damage', y='AP50', title='Average Precision (AP) vs. Damage Level')
    fig.write_html('damage_experiment.html', auto_open=True)
        
        
    
        
        
    
    
    
    
    
