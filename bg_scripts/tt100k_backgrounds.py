"""Script to create backgrounds for the synthetic dataset generator from the
TT100K dataset. It filters out and saves TT100K images that have an
instance of a front-facing traffic sign.
"""

def main():
    import os
    import argparse
    from glob import glob
    import json
    import shutil

    parser = argparse.ArgumentParser(description='Filter TT100K images to have no front-facing traffic signs.')
    parser.add_argument('--gt', type=str, help='Complete path to directory containing TT100K ground truth',
                        default='D:/Datasets/TT100K_2016/data')
    args = parser.parse_args()

    def save_images():
        signs = 0
        saved = 0
        count = 0  ##
        with open(f"{args.gt}{os.sep}annotations.json", 'r') as f:
            gt_json = json.load(f)
        for img_id in gt_json['imgs']:
            if len(gt_json['imgs'][img_id]['objects']) != 0:
                signs += 1
                count += len(gt_json['imgs'][img_id]['objects'])
            elif gt_json['imgs'][img_id]['path'].split('/')[0] == 'other':
                continue
            else:
                img_fn = os.path.join(os.path.abspath(args.gt), gt_json['imgs'][img_id]['path'])
                if os.path.exists(img_fn):
                    # Save image to new directory
                    img_fn_new = os.path.join(os.path.abspath(args.gt), 'backgrounds', f'{img_id}.jpg')
                    os.makedirs(os.path.dirname(img_fn_new), exist_ok=True)
                    shutil.copyfile(img_fn, img_fn_new)
                    print(f"Saved {img_fn_new}")
                    saved += 1
                else:
                    print(f"{img_fn} does not exist")
        print(f"Skipped {signs} images with traffic signs")
        print(f"Skipped {count} signs")  ##
        print(f"Saved {saved} images without traffic signs")

    save_images()

if __name__ == "__main__":
    main()
