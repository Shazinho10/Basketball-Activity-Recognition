from ultralytics import YOLO
from IPython.display import display, Image
import cv2
import torch
import torchvision.ops.boxes as bops
import imutils
import matplotlib.pyplot as plt
from shapely.geometry import Polygon




class ROI:
    def __init__(self, model) -> None:
        self.model = model

    def hoop_coordinates(self, image):
        results = self.model.predict(source=image, conf=0.25)
        coordinates = results[0].boxes.xyxy
        classes = results[0].boxes.cls
        classes = torch.tensor(classes, dtype=torch.int32)

        x1, y1, x2, y2 = 0,0,0,0
        for cord, cls in zip(coordinates, classes):
            if cls == 2:
                x1 = int(cord[0])
                y1 = int(cord[1])
                x2 = int(cord[2])
                y2 = int(cord[3])
        bbox = {"x1":x1, "y1":y1, "x2":x2, "y2":y2}
        return bbox
    

    def ball_coordinates(self, image):
        results = self.model.predict(source=image, conf=0.25)
        if len(results) > 1:
            print("-"*20)

        coordinates = results[0].boxes.xyxy
        classes = results[0].boxes.cls
        classes = torch.tensor(classes, dtype=torch.int32)

        x1, y1, x2, y2 = 0,0,0,0
        for cord, cls in zip(coordinates, classes):
            if cls == 1:
                x1 = int(cord[0])
                y1 = int(cord[1])
                x2 = int(cord[2])
                y2 = int(cord[3])
        bbox = {"x1":x1, "y1":y1, "x2":x2, "y2":y2}
        return bbox
    
    
    def player_coordinates(self, image):   #returns a dictionary of players number and the bounding box | each key has [x1, y1, x2, y2]
        results = self.model.predict(source=image, conf=0.25)
        coordinates = results[0].boxes.xyxy
        classes = results[0].boxes.cls
        classes = torch.tensor(classes, dtype=torch.int32)

        count = 0
        player_no = {}
        for cord, cls in zip(coordinates, classes):
            if cls == 0:
                count += 1
                x1 = int(cord[0])
                y1 = int(cord[1])
                x2 = int(cord[2])
                y2 = int(cord[3])

                if count not in player_no:
                    player_no[count] = []
                player_no[count] = [x1, y1, x2, y2]

        return player_no
    

    def calculate_iou(self, bbox1, bbox2):
        '''
        bbox1 is a dict containing [x1,y1,w,h]
        bbox2 is a dict containing [x1,y1,w,h]
        '''


        # bbox1_x2 = bbox1['x2'] - bbox1['x1'] 
        # bbox1_y2 = bbox1['y2'] - bbox1['y1']
        # bbox2_x2 = bbox2['x2'] - bbox2['x1']
        # bbox2_y2 = bbox2['y2'] - bbox2['y1']


        bbox1_x2 = bbox1['x1'] + bbox1['x2']
        # bbox1_x2 = bbox1_x2 // 2
        bbox1_y2 = bbox1['y1'] + bbox1['y2']
        # bbox1_y2 = bbox1_y2 // 2
        bbox2_x2 = bbox2['x1'] + bbox2['x2']
        # bbox2_x2 = bbox2_x2 // 2
        bbox2_y2 = bbox2['y1'] + bbox2['y2']
        # bbox2_y2 = bbox2_y2 // 2

        box1 = torch.tensor([[bbox1['x1'], bbox1['y1'], bbox1_x2, bbox1_y2]], dtype=torch.float)
        box2 = torch.tensor([[bbox2['x1'], bbox2['y1'], bbox2_x2, bbox2_y2]], dtype=torch.float)
        # box1 = torch.tensor([[bbox1['x1'], bbox1['y1'], bbox1['x2'], bbox1['y2']]], dtype=torch.float)
        # box2 = torch.tensor([[bbox2['x1'], bbox2['y1'], bbox2['x2'], bbox2['y2']]], dtype=torch.float)
        iou = bops.box_iou(box1, box2)
        return iou.item()
    
    
    
    def bb_intersection_over_union(self, bbox1, bbox2):
        boxA = []
        boxB = []
        for key in bbox1:
            boxA.append(bbox1[key])
        
        for key in bbox2:
            boxB.append(bbox2[key])

        # determine the (x, y)-coordinates of the intersection rectangle        
        detection = Polygon([(boxA[0], boxA[1]), (boxA[0], boxA[3]), (boxA[2], boxA[3]), (boxA[2], boxA[1])])
        region = Polygon([(boxB[0], boxB[1]), (boxB[0], boxB[3]), (boxB[2], boxB[3]), (boxB[2], boxB[1])])
        intersect = detection.intersection(region).area
        union = detection.union(region).area
        a = 39
        if union != 0.0:
            return int(intersect / union)

# if __name__ == "__main__":
    import numpy as np

    # template = np.full((1080, 1280, 3), (255, 255, 255), dtype="uint8")
    
    # box1 = {'x1': 891, 'x2': 922, 'y1': 261, 'y2': 291}
    # box2 = {'x1': 740, 'x2': 801, 'y1': 206, 'y2': 279}
    # cv2.rectangle(template, (box1["x1"], box1["y1"]), (box1["x2"], box1["y2"]))
    # ious = calculate_iou(box1, box2)
    # print(ious)
