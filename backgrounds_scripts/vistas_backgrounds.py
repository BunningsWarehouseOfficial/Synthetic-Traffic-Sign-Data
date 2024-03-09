"""Script to create backgrounds for the synthetic dataset generator from the
Mapillary Vistas dataset. It filters out and saves Vistas images that have an
instance of a front-facing traffic sign.
"""

def main():
    import os
    import argparse
    from glob import glob
    import json
    import shutil

    parser = argparse.ArgumentParser(description='Filter Vistas images to have no front-facing traffic signs.')
    parser.add_argument('--gt', type=str, help='Complete path to directory containing Vistas ground truth',
                        default='D:/Datasets/Vistas')
    args = parser.parse_args()

    def save_images(mode='training'):
        gt_dir = os.path.join(os.path.abspath(args.gt), mode, 'v2.0', 'polygons')
        gt_jsons = glob(f"{gt_dir}{os.sep}*.json", recursive=True)
        signs = 0
        saved = 0
        for gt_json_fn in gt_jsons:  # Iterate through each annotation file
            # Check image-wise json annotations for traffic signs
            with open(gt_json_fn, 'r') as f:
                gt_json = json.load(f)
            traffic_sign = False
            filtered_objects = {
                'object--traffic-sign--front',
                'object--traffic-sign--temporary-front',
                'object--traffic-sign--information-parking',
                'object--traffic-sign--direction-front',
                'object--traffic-sign--ambiguous'
            }
            for o in gt_json['objects']:
                if o['label'] in filtered_objects:
                    signs += 1
                    traffic_sign = True
                    break
            if not traffic_sign:
                img_fn = gt_json_fn.replace(f'v2.0{os.sep}polygons', 'images').replace('.json', '.jpg')
                if os.path.exists(img_fn):
                    # Save image to new directory
                    img_fn_new = img_fn.replace(f'{os.sep}{mode}{os.sep}images', f'{os.sep}{mode}_backgrounds')
                    os.makedirs(os.path.dirname(img_fn_new), exist_ok=True)
                    shutil.copyfile(img_fn, img_fn_new)
                    print(f"Saved {img_fn_new}")
                    saved += 1
                else:
                    print(f"{img_fn} does not exist")
        print(f"Skipped {signs} {mode} images with traffic signs")
        print(f"Saved {saved} {mode} images without traffic signs")

    save_images('training')
    save_images('validation')

if __name__ == "__main__":
    main()
