from ultralytics import YOLO
from IPython.display import display, Image
import cv2
import torch
import torchvision.ops.boxes as bops
import imutils
import matplotlib.pyplot as plt
from roi import ROI
import numpy as np
from field_classification import shot_classification
import os

model = YOLO("Detections/ultralytics/runs/detect/train5/weights/best.pt")
roi = ROI(model)
CLASSES = ["background", "3point_Region", "freethrow_circle", "2point_region"]   #different regions of the basketball court 
ball_hoop_threshold = 0.4   #this is the hyper parameter that can be handled further

#testing the max IOU
all_ball_hoop_ious = []    #storing the iou for ball and hoop in every frame
posession_ious = []    #storing IOU for ball and players of every frame
frame_no = 0
frame_max_iou = {}  #storing the frame number and the max IOU
frame_id_iou = {} #storing the frame number player id and the IOU
frame_id_cord = {}  #storing the frame ids and the coordinates of the player bbox

frames = {}

test_folder = "testing/three_point"

# predictions = []
# for i in os.listdir(test_folder):
# full_path = os.path.join(test_folder, i)

cap = cv2.VideoCapture("testing/two_point/two_1_c_2.mp4")

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while(cap.isOpened()):
  frame_no += 1
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    frame = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    bbox1 = roi.ball_coordinates(frame)
    bbox2 = roi.hoop_coordinates(frame)
    ball_hoop_iou = roi.calculate_iou(bbox1, bbox2)

    if ball_hoop_iou >= 0.3:
       a = 34
    all_ball_hoop_ious.append(ball_hoop_iou)


    ################################### Test Visualisation ####################################################
    results = model.predict(source=frame, conf=0.25)
    coordinates = results[0].boxes.xyxy
    classes = results[0].boxes.cls
    classes = torch.tensor(classes, dtype=torch.int32)
    visframe = frame.copy()
    for cord, cls in zip(coordinates, classes):
      x1 = int(cord[0])
      y1 = int(cord[1])
      x2 = int(cord[2])
      y2 = int(cord[3])
      visframe = cv2.rectangle(visframe, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    
    cv2.imshow('Frame',visframe)
    # Press Q on keyboard to  exit
    if cv2.waitKey(20) & 0xFF == ord('q'):
      break




  ################################### Test Visualisation ####################################################

    #getting all of the frames of the players
    player_boxes = roi.player_coordinates(frame)

    if frame_no not in frame_id_iou:
        frame_id_iou[frame_no] = {}

    if frame_no not in frame_id_cord:
        frame_id_cord[frame_no] = {}

    if frame_no not in frames:
        frames[frame_no] = np.zeros(frame.shape)
    frames[frame_no] = frame

    player_id = 0
    for i in player_boxes:
        x1 = player_boxes[i][0]
        y1 = player_boxes[i][1]
        x2 = player_boxes[i][2]
        y2 = player_boxes[i][3]
        

        player_cords = {"x1":x1, "y1":y1, "x2":x2, "y2":y2}
        ball_player_iou = roi.calculate_iou(player_cords, bbox1)
        posession_ious.append(ball_player_iou)
        
        
        if player_id not in frame_id_iou[frame_no]:
          frame_id_iou[frame_no][player_id] = 0
        frame_id_iou[frame_no][player_id] = ball_player_iou  #storing the frame number, player id and the IOU of ball and player.

        
        if player_id not in frame_id_cord[frame_no]:
          frame_id_cord[frame_no][player_id] = {}
        frame_id_cord[frame_no][player_id] = player_cords  #storing frame number, player id and the player coordinates

        player_id += 1      #assigning an integer id to each player
    
    if frame_no not in frame_max_iou:
      frame_max_iou[frame_no] = 0
    frame_max_iou[frame_no] = max(posession_ious)   #each frame's max IOU

    #storing frames against their number

  
  else: 
    break
  
cap.release()
cv2.destroyAllWindows()
############################################# Start of The Hueristic Logic ###########################################

if max(all_ball_hoop_ious) >= ball_hoop_threshold:
    max_iou_frame = max(frame_max_iou, key=lambda x:frame_max_iou[x])   #getting the frame where max IOU was stored
    max_iou_player = max(frame_id_iou[max_iou_frame], 
                        key=lambda x:frame_id_iou[max_iou_frame][x])  #getting the player id with max IOU
    
    shot_cords = frame_id_cord[max_iou_frame][max_iou_player]
    shot_frame = frames[max_iou_frame]
    shot_frame = cv2.rectangle(shot_frame, (shot_cords["x1"],shot_cords["y1"]),
                                (shot_cords["x2"], shot_cords["y2"]), (255, 0, 0), 2)
    
    shot_region = shot_classification(shot_frame, shot_cords).item()
    print(f"shot type is {CLASSES[shot_region]}")


      ############################################visualizing the frame where shot is being taken ###############################################
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(shot_frame              , 
                CLASSES[shot_region], 
                (50, 50), 
                font, 1,  
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
    
    cv2.imshow('shot_Frame',shot_frame)
    cv2.waitKey(0)
      
      ###########################################################################################################################################

    # a = 10
    # with open('three_point.txt', 'a') as f:
    #   f.write(str(shot_region))
    #   f.write('\n')
        
      
    
    

    