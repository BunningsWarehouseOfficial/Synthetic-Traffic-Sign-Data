"""Script for generating sequences of traffic sign detection training data. Imitates how a moving camera would capture
multiple frames of a scene where the target object is at slightly different distances from the camera in each frame.
"""
# Author: Kristian Rados

OUT_DIR     = "SGTSD_Sequences"
LABELS_FILE = "labels.txt"
import argparse
from datetime import datetime
import os
from utils import load_paths, load_files, resize, overlay


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("bg_dir", type=dir_path, help="path to background image directory")
    parser.add_argument("fg_dir", type=dir_path, help="path to foreground/sign image directory")
    return parser.parse_args()

def dir_path(path):
    """Validate an argument type as being a directory path."""
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"""{path} is not a valid path""")


def main():
    args = parse_arguments()
    labels_path = os.path.join(OUT_DIR, LABELS_FILE)

    # Directory structure setup
    if (not os.path.exists(OUT_DIR)):
        out_dir = OUT_DIR
    else:
        timestamp = datetime.now().strftime("_%d%m%Y_%H%M%S")
        out_dir = OUT_DIR + timestamp
    os.mkdir(out_dir)

    # Loading argument-specified directories
    bg_paths = load_paths(args.bg_dir)
    fg_paths = load_paths(args.fg_dir)
    num_bg = len(bg_paths)
    num_fg = len(fg_paths)
    print(f"Found {num_bg} background images.")
    print(f"Found {num_fg} foreground images.")
    if num_bg == 0 or num_fg == 0:
        print("Error: each input directory must have at least 1 image")
        return

if __name__ == "__main__":
    main()