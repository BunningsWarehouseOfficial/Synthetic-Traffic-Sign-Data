import os
import cv2
import argparse
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from damage_eval import BoundingBox, get_detection_metrics

current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/home/allenator/Pawsey-Internship/datasets/SGTS_Dataset')
parser.add_argument('--gt_file', default='/home/allenator/Pawsey-Internship/datasets/SGTS_Dataset/_single_annotations_array.npy')
parser.add_argument('--eval_file', default='/home/allenator/Pawsey-Internship/eval_dir/SGTS_Dataset_dmg_assess/SGTS_Dataset_efficientdet-d2.npy')
parser.add_argument('--out_dir', default='.')


def get_bounding_boxes(gt_detections, pred_detections):
    # gt detections array in format [image_id, xtl, ytl, width, height, damage_1, damage_2, ..., damage_n, class_id]
    gt_boxes = [BoundingBox(image_id=det[0], class_id=det[-1], xtl=det[1], ytl=det[2], xbr=det[1] + det[3], ybr=det[2] + det[4], 
                        damages=det[5:-1]) for det in gt_detections]
    # pred detections array in format [image_id, xtl, ytl, width, height, score, damage_1, damage_2, ..., damage_n, class_id]
    pred_boxes = [BoundingBox(image_id=det[0], class_id=det[-1], xtl=det[1], ytl=det[2], xbr=det[1] + det[3], ybr=det[2] + det[4], 
                              score=det[5], damages=det[6:-1]) for det in pred_detections]
    return gt_boxes, pred_boxes


if __name__ == '__main__':
    args = parser.parse_args()
    gt_arr = np.load(args.gt_file)
    pred_arr = np.load(args.eval_file)
    gt_boxes, pred_boxes = get_bounding_boxes(gt_arr, pred_arr)
    metrics = get_detection_metrics(gt_boxes, pred_boxes, dmg_threshold=0.1)
    
    x_vals = metrics.interpolated_recall
    y_vals = metrics.interpolated_precision
    
    plt.plot(x_vals, y_vals, label='Precision vs Recall for threshold = 3')
    plt.ylabel('Interpolated Precision')
    plt.xlabel('Interpolated Recall')
    plt.show()
    
    print(metrics.ap)
    
    