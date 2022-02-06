import os
import cv2
import argparse
import json
import numpy as np

from damage_experiment import split_by_damage

current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/home/allenator/Pawsey-Internship/datasets/sgts_sequences')
parser.add_argument('--gt_anns', default='/home/allenator/Pawsey-Internship/datasets/sgts_sequences/_annotations.coco.json')
parser.add_argument('--gt_file', default='/home/allenator/Pawsey-Internship/datasets/sgts_sequences/_single_annotations_array.npy')
parser.add_argument('--eval_file', default='/home/allenator/Pawsey-Internship/eval_dir/sgts_sequences/0.1_augmented_efficientdet-d2.npy')
parser.add_argument('--out_dir', default='.')
    

if __name__ == '__main__':
    args = parser.parse_args()    
    
    gt_arr = np.load(args.gt_file)
    pred_arr = np.load(args.eval_file)
    with open(args.gt_anns, 'r') as f:
        gt_anns = json.load(f)
    
    split_arr, damages = split_by_damage(gt_arr, pred_arr)
    for i, dmg_arr in enumerate(split_arr):
        closest_images = dmg_arr[np.nonzero((dmg_arr[:, 0] + 1) % 8 == 0)]
        if len(closest_images) == 0: continue
        top_pred = closest_images[np.argmax(closest_images[:, 5])]
        id = top_pred[0]
        xtl, ytl, width, height = top_pred[1:5].astype(np.int32)
        f_path = os.path.join(args.dataset_dir, gt_anns['images'][int(id)]['file_name'])
        image = cv2.imread(f_path)
        image = cv2.rectangle(image, (xtl, ytl), (xtl+width, ytl+height), color=(255,234,43))
        image = cv2.putText(image, '{:.2f}'.format(top_pred[5]), (xtl, ytl-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,234,43), 2)
        cv2.imwrite(os.path.join(args.out_dir, f'{damages[i]}_damage.jpg'), image)
        
        
        
        
        
        
    