import sys
from collections import Counter, defaultdict
from typing import List, Dict

import numpy as np

from detection_eval import Box, BoundingBox, MetricPerClass, calculate_all_points_average_precision


def get_AP_metrics(gold_standard: List[BoundingBox],
                           predictions: List[BoundingBox],
                           dmg_threshold: float = 0.2, 
                           num_sectors: int = 4) -> Dict[str, MetricPerClass]:
    """
    Args:
        gold_standard: ground truth bounding boxes;
        predictions: detected bounding boxes;
        iou_threshold: IOU threshold indicating which detections will be considered TP or FP (default value = 0.5);
        dmg_threshold: required damage level for sector to be marked as 'damaged';
        num_sectors: number of sectors in the damage assessment;
    Returns:
        A dictionary containing metrics of each class.
    """
    ret = {}  # list containing metrics (precision, recall, average precision) of each class

    # Get all classes
    classes = sorted(set(b.class_id for b in gold_standard))

    # Precision x Recall is obtained individually by each class
    # Loop through by classes
    for c in classes:
        preds = [b for b in predictions if b.class_id == c]  # type: List[BoundingBox]
        golds = [b for b in gold_standard if b.class_id == c]  # type: List[BoundingBox]
        npos = len(golds) * num_sectors # number of ground truth damages

        # sort detections by decreasing confidence
        preds = sorted(preds, key=lambda b: b.score, reverse=True)
        tps = np.zeros((len(preds), num_sectors))
        fps = np.zeros((len(preds), num_sectors))

        # create dictionary with amount of gts for each image
        counter = Counter([cc.image_id for cc in golds])
        for key, val in counter.items():
            counter[key] = np.zeros(val)

        # Pre-processing groundtruths of the some image
        image_id2gt = defaultdict(list)
        for b in golds:
            image_id2gt[b.image_id].append(b)
            
        tp_IOUs = []
        tp_scores = []
        # Loop through detections
        for i in range(len(preds)):
            # Find ground truth image
            gt = image_id2gt[preds[i].image_id]
            max_iou = sys.float_info.min
            mas_idx = -1
            for j in range(len(gt)):
                iou = Box.intersection_over_union(preds[i], gt[j])
                if iou > max_iou:
                    max_iou = iou
                    mas_idx = j
                  
            if counter[preds[i].image_id][mas_idx] == 0:
                # Add IOU of best detection for this ground truth
                tp_IOUs.append(max_iou)
                # Add score of best detection for this ground truth
                tp_scores.append(preds[i].score)
                
                pred_damages = np.array([1 if d >= dmg_threshold else 0 for d in preds[i].damages])
                gt_damages = np.array([1 if d >= dmg_threshold else 0 for d in gt[mas_idx].damages])
                tps[i, :] = (pred_damages == gt_damages).astype(np.uint8)
                fps[i, :] = (pred_damages != gt_damages).astype(np.uint8)
                counter[preds[i].image_id][mas_idx] = 1  # flag as already 'seen'
                
        # compute precision, recall and average precision
        tps = np.reshape(tps, -1)
        fps = np.reshape(fps, -1)
        cumulative_fps = np.cumsum(fps)
        cumulative_tps = np.cumsum(tps)
        recalls = np.divide(cumulative_tps, npos, out=np.full_like(cumulative_tps, np.nan), where=npos != 0)
        precisions = np.divide(cumulative_tps, (cumulative_fps + cumulative_tps))
        # Depending on the method, call the right implementation
        ap, mpre, mrec, _ = calculate_all_points_average_precision(recalls, precisions)
        # add class result in the dictionary to be returned
        r = MetricPerClass()
        r.class_id = c
        r.precision = precisions
        r.recall = recalls
        r.ap = ap
        r.interpolated_recall = np.array(mrec)
        r.interpolated_precision = np.array(mpre)
        r.tp = np.sum(tps)
        r.fp = np.sum(fps)
        r.num_groundtruth = len(golds)
        r.num_detection = len(preds)
        r.tp_IOUs = tp_IOUs
        r.tp_scores = tp_scores
        ret[c] = r
        
    if len(ret.keys()) == 1:
        ret = list(ret.values())[0]
    return ret