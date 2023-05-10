import numpy as np
import cv2
import torch
import torchvision.transforms as T
from segmentation_handler import get_seg_map


def rescale_coordinate(x, pct):
    x = x - int((x*pct)/100)
    return x


transform = T.Resize((1089, 1921))

def shot_classification(image, cords):   #for the IOU approach
    seg_map = get_seg_map(image)
    seg_map = transform(seg_map)
    x1, y1, x2, y2 = cords["x1"], cords["y1"], cords["x2"], cords["y2"]
    cls = seg_map[:, y2, (x1 + x2)//2]
    return cls



def shot_coordinate(image, feet):   # for the keypoint approach
    seg_map = get_seg_map(image)
    seg_map = transform(seg_map)
    cls = seg_map[:, feet[1], feet[0]]
    return cls

def get_seg_transform(image):
    seg_map = get_seg_map(image)
    seg_map = transform(seg_map)
    return seg_map

# if __name__ == "__main__":
#     image = cv2.imread("testing/test_images/00070200.jpg")
#     cls = shot_classification(image)
#     print(cls)