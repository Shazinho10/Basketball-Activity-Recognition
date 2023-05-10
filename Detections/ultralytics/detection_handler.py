from ultralytics import YOLO
from IPython.display import display, Image
import cv2
import torch
import imutils
import matplotlib.pyplot as plt
from trackers.adlytic_tracker import Adlytic_YoloX_CPP_ByteTrack as b_tracker


model = YOLO("best.pt")

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
    # image = cv2.imread("test_image.jpg")
    # x1, y1, x2, y2 = ball_coordinates(image)
    # image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # image = imutils.resize(image, width=640)
    # cv2.imshow("ball frame", image)
    # cv2.waitKey(0)	
    import cv2
    import numpy as np
    tracker = b_tracker()
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('video 1.mp4')
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            results = model.predict(source=frame, conf=0.25)
            coordinates = results
            # ids =tracker.apply(coordinates)
            for cord in results[0].boxes.xyxy:
                x1 = int(cord[0])
                y1 = int(cord[1])
                x2 = int(cord[2])
                y2 = int(cord[3])
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.imshow('Frame',frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break
        
        # When everything done, release the video capture object
                