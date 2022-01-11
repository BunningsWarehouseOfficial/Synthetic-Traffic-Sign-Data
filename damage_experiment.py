import os
import argparse
import numpy as np
import pandas as pd
import plotly.express as px

from collections import defaultdict
from evaluation_metrics.detection_eval import BoundingBox, get_pascal_voc_metrics, Box


current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--gt_file', default='/home/allenator/Pawsey-Internship/datasets/sgts_sequences/_single_annotations_array.npy', 
                    help='Ground truth annotations for dataset as a numpy file')
parser.add_argument('--eval_file', default='/home/allenator/Pawsey-Internship/eval_dir/sgts_sequences/efficientdet-d0.npy', 
                    help='File containing evaluated detections as a numpy file')
parser.add_argument('--num_frames', default=8, type=int, help='Number of frames per sequence in dataset')
parser.add_argument('--experiment', default='damage', choices=['damage', 'distance'] , help='Type of experiment to evaluate')


class SequenceEvaluation:
    def __init__(self, gt_boxes, pred_boxes):
        self.pred_boxes = pred_boxes
        self.gt_boxes = sorted(gt_boxes, key=lambda x: x.image_id)
    
    # Proposed heuristic 1: use the bounding box with the maximum score for each image sequence.    
    def score_heuristic(self):
        """
        Choose the detection among the ones generated for this sequence, which has the highest score.
        """
        det = self.pred_boxes[np.argmax([bbox.score for bbox in self.pred_boxes])]
        [ann] = list(filter(lambda x: x.image_id == det.image_id, self.gt_boxes))
        return Box.intersection_over_union(det, ann)
    
    # Proposed heuristic 2: use the bounding box with area closest to width ** 2
    def area_heuristic(self, gt_width):
        """
        Choose the detection among the ones generated for this sequence which has size closest to the optimal width.
        """
        area_diffs = []
        gt_area = gt_width ** 2
        for det in self.pred_boxes:
            det_area = (det.xbr - det.xtl) * (det.ybr - det.ytl)
            area_diffs.append(abs(det_area - gt_area))
        optimal_det = self.pred_boxes[np.argmin(area_diffs)]
        [ann] = list(filter(lambda x: x.image_id == optimal_det.image_id, self.gt_boxes))
        return Box.intersection_over_union(optimal_det, ann)
        
                
def get_metrics(gt, pred):
    """
    Calculates the metrics for a given set of ground truth and predicted detections.
    Metrics in the format [AP50, AP75, AP95, max precision, max recall, min precision, min recall]
    """   
    columns = ['AP50', 'mAP', 'Mean IOU', 'Mean Score']
    metrics = np.zeros(len(columns))
    AP40_metrics = get_pascal_voc_metrics(gt, pred, iou_threshold=0.4)
    tp_IOUs = AP40_metrics.tp_IOUs
    tp_scores = AP40_metrics.tp_scores
    APs = []
    for threshold in np.arange(0.5, 1.0, 0.05):
        APs.append(get_pascal_voc_metrics(gt, pred, iou_threshold=threshold).ap)
    metrics[0] = APs[0]
    metrics[1] = np.mean(APs)
    metrics[2] = np.mean(tp_IOUs)
    metrics[3] = np.mean(tp_scores)
    return metrics, columns


def get_bounding_boxes(gt_detections, pred_detections):
    # gt detections array in format [image_id, bg_id, fg_id, xtl, ytl, width, height, damage, distance, class_id]
    gt_boxes = [BoundingBox(image_id=det[0], class_id=det[-1], xtl=det[3], ytl=det[4], xbr=det[3] + det[5], ybr=det[4] + det[6]) 
                        for det in gt_detections]
    # pred detections array in format [image_id, xtl, ytl, width, height, score, class_id]
    pred_boxes = [BoundingBox(image_id=det[0], class_id=det[-1], xtl=det[1], ytl=det[2], xbr=det[1] + det[3], ybr=det[2] + det[4], score=det[5])
                    for det in pred_detections]
    return gt_boxes, pred_boxes


def prune_detections(detections_array, max_detections=50):
    sorted_indices = np.argsort(detections_array[:, 5], axis=0)
    pruned_arr = detections_array[sorted_indices][::-1]
    return pruned_arr[:max_detections]


def split_by_sign_size(gt_array, arr_to_split):
    gt_array = gt_array[np.argsort(gt_array[:, 0])] # Sort gt_array by image_id just in case
    split_dict = defaultdict(list)
    
    for i in range(len(arr_to_split)):
        image_id = arr_to_split[i, 0]
        size = gt_array[int(image_id), 5]   # sign width in pixels
        split_dict[size].append(arr_to_split[i])
            
    split_dict = {dist:np.array(arr) for dist, arr in split_dict.items()}
    return split_dict


def split_by_sequence(arr, num_frames):
    """
    Splits an array of detections into sequences of detections.
    """
    split_indices = []
    # Sort array by image_id
    arr = arr[np.argsort(arr[:, 0])]
    image_ids = arr[:, 0]
    for i in range(len(image_ids) - 1):
        id = int(image_ids[i])
        is_boundary_index = image_ids[i - 1] != image_ids[i]  # Check if image_id has changed
        if id % num_frames == 0 and id != 0 and is_boundary_index:
            split_indices.append(i)
    split_arrs = np.split(arr.copy(), split_indices)
    return np.array(split_arrs)  


def metrics_by_size(gt_array, pred_array):
    sizes_gt = split_by_sign_size(gt_array, gt_array)
    max_detections = int(np.mean([len(arr) for arr in sizes_gt.values()]) * 5)
    sizes_pred = split_by_sign_size(gt_array, pred_array)
    sizes_pred = {dist:prune_detections(arr, max_detections) for dist, arr in sizes_pred.items()}
    metrics_array = None
    
    # Format [Sign Width, AP50, mAP, Maximum IOU, Minimum IOU, Mean IOU, Maximum Score, Minimum Score, Mean Score]
    
    for size in sizes_gt:
        gt_boxes, pred_boxes = get_bounding_boxes(sizes_gt[size], sizes_pred[size])
        metrics, columns = get_metrics(gt_boxes, pred_boxes)
        row = np.zeros((1, len(metrics) + 1))
        row[0, 0] = size
        row[0, 1:] = metrics
        if metrics_array is None:
            metrics_array = row
        else:
            metrics_array = np.append(metrics_array, row, axis=0)
    columns = ['Sign Width'] + columns
    return metrics_array, columns 


def metrics_by_sequence(gt_array, pred_array):
    # Format [Sign Width, AP50, mAP, Maximum IOU, Minimum IOU, Mean IOU, Maximum Score, Minimum Score, Mean Score]
    size_metrics, _ = metrics_by_size(gt_array, pred_array)
    optimal_width = int(size_metrics[np.argmax(size_metrics[:, 2]), 0])
    closest_width = np.max(size_metrics[:, 0])
    
    sequences_gt = split_by_sequence(gt_array, args.num_frames)
    sequences_pred = split_by_sequence(pred_array, args.num_frames)
    sequences_pred = [prune_detections(arr) for arr in sequences_pred]
    
    # Format [Damage, AP50, mAP, Maximum IOU, Minimum IOU, Mean IOU, Maximum Score, Minimum Score, Mean Score] 
    # for each sequence
    metrics_array = None
    
    # iterate over each image sequence
    for i in range(len(sequences_gt)):
        damage_level = round(np.mean(sequences_gt[i][:, 7]), 1)
        gt_boxes, pred_boxes = get_bounding_boxes(sequences_gt[i], sequences_pred[i])
        metrics, columns = get_metrics(gt_boxes, pred_boxes)
        
        row = np.zeros((1, len(metrics) + 3))
        row[0, 0] = damage_level
        eval = SequenceEvaluation(gt_boxes, pred_boxes)
        row[0, 1:3] = np.array([eval.score_heuristic(), eval.area_heuristic(optimal_width)])
        row[0, 3:] = metrics
        
        if metrics_array is None:
            metrics_array = row
        else:
            metrics_array = np.append(metrics_array, row, axis=0)
    columns = ['Damage', 'Score Heuristic IOU', 'Area Heuristic IOU'] + columns
    return metrics_array, columns


def damage_experiment(gt_array, pred_array):
    metrics_array, columns = metrics_by_sequence(gt_array, pred_array) 
    
    damages = np.unique(metrics_array[:, 0])
    dmg_metrics = np.zeros((len(damages), metrics_array.shape[1]))
    for i, dmg in enumerate(damages):
        dmg_arr = metrics_array[np.nonzero(metrics_array[:, 0] == dmg)]
        dmg_metrics[i] = np.mean(dmg_arr, axis=0)
    return pd.DataFrame(dmg_metrics, columns=columns)


def distance_experiment(gt_array, pred_array):
    metrics_array, columns = metrics_by_size(gt_array, pred_array)
    return pd.DataFrame(data=metrics_array, columns=columns)
    

if __name__ == '__main__':
    args = parser.parse_args()
    gt_array = np.array(np.load(args.gt_file), dtype=np.float32)
    pred_array = np.array(np.load(args.eval_file), dtype=np.float32)
    
    # A plot of a metric (e.g., mAP) against various damage average levels (e.g., 10%, 20%, etc.), where AP is evaluated
    # across a sequence, and metrics are averaged across sequences with the same average damage level.
    
    # Findings: it has been found that metrics remain consistent until a threshold damage level, 50%, is reached, whereupon
    # model performance reduces rapidly as damage increases.
    if args.experiment == 'damage':
        df = damage_experiment(gt_array, pred_array)
        df_long = pd.melt(df, id_vars=['Damage'], value_vars=['Mean IOU', 'Score Heuristic IOU', 'Area Heuristic IOU'])
        fig = px.line(df_long, x='Damage', y='value', title='IOU vs. Damage Level', color='variable')
        
    
    # A plot of a metric (e.g., mAP) against width of the sign in the image, where width represents 'distance' 
    # from the sign to the camera. I.e., closer signs have higher pixel width.
    
    # Findings: it has been found that mAP reaches a maximum at a width of ~40 pixels, and decreases until ~55 pixels,
    # before plateauing. 
    elif args.experiment == 'distance':
        df = distance_experiment(gt_array, pred_array)
        print(df)
        fig = px.line(df, x='Sign Width', y='mAP', title='Average Precision (AP) vs. width of sign in pixels in image')
    fig.show()
    
    
    
    
        
        
    
        
        
    
    
    
    
    
