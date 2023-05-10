import os
import shutil
import argparse


parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('--image_folder', type=str,
                    help='A required string argument')
parser.add_argument("--annotation_folder", type=str, help="A required string argument")

args = parser.parse_args()

train_size = int(len(os.listdir(args.image_folder))*0.9)

for i,slice in enumerate(os.listdir(args.image_folder)):
    if i < train_size:
          shutil.copy(args.image_folder + slice, "data/images/training/"+slice)
          shutil.copy(args.annotation_folder +slice.split(".")[0] + ".png", "data/annotations/training/" + slice.split(".")[0] + ".png")
        
    elif i >= train_size:
          shutil.copy(args.image_folder + slice, "data/images/validation/"+slice)
          shutil.copy(args.annotation_folder +slice.split(".")[0] + ".png", "data/annotations/validation/" + slice.split(".")[0] + ".png")
              
