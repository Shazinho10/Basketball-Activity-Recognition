from ultralytics import YOLO
from IPython.display import display, Image
import cv2
import torch
import imutils
import matplotlib.pyplot as plt


model = YOLO("Detections/ultralytics/runs/detect/train5/weights/best.pt")

def ball_coordinates(image):
    results = model.predict(source=image, conf=0.25)
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
    return x1, y1, x2, y2

if __name__ == "__main__":
    image = cv2.imread("testing/test_images/00070200.jpg")
    # x1, y1, x2, y2 = ball_coordinates(image)
    # image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # image = imutils.resize(image, width=640)
    # cv2.imshow("ball frame", image)
    # cv2.waitKey(0)