from ultralytics import YOLO
from IPython.display import display, Image
import cv2
import torch
import torchvision.ops.boxes as bops
import imutils
import matplotlib.pyplot as plt
from roi import ROI
import numpy as np
from utils.datasets import letterbox
from field_classification import shot_coordinate, get_seg_transform
# from field_classification import shot_classification
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import os
import time
from torchvision import transforms
from multipose_handler import get_keypoints
import math


#################################### Warmup of the Models ################
vid_path = "testing_videos/heuristic_testing_data/three_point/three_25_c_2.mp4"
device = "cuda" if torch.cuda.is_available() else "cpu"

#YOLOV8 Detector
model = YOLO("detection_weights/best_30_march.pt")
roi = ROI(model)

#YOLOv7 Multi Pose Estimator
pose_weigths = torch.load('yolov7-w6-pose.pt')
pose_model = pose_weigths['model']
pose_model = pose_model.half().to(device)
_ = pose_model.eval()

CLASSES = ["background", "3point_Region", "freethrow_circle", "2point_region"]   #different regions of the basketball court 
ball_hoop_threshold = 0.35   #this is the hyper parameter that can be handled further
ball_hand_threshold = 10     #determines how close the ball is to the hand bounding box

#testing the max IOU
all_ball_hoop_ious = []    #storing the iou for ball and hoop in every frame

frame_no = 0
frame_max_iou = {}  #storing the frame number and the max IOU
frame_ball_cord = {}
frame_hand_keypoints = {} #storing all of the keypoints of hands against the frame number
frame_feet_keypoints = {} #storing all of the keypoints of feet against the frame number
frame_knee_keypoints = {} #storing the knee keypoints

frames = {}
frame_bank = []

cap = cv2.VideoCapture(vid_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.
result = cv2.VideoWriter('test.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, (1920,1088))
while(cap.isOpened()):


  ret, frame = cap.read()
  if ret == True:
    orig_image = frame
    # result.write(frame)
    detect_frame = letterbox(frame, (frame_width), stride=64, auto=True)[0]

    #converting the frame according to the pose estimation model
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = letterbox(image, (frame_width), stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    image = image.half()

    # Get the start time.
    start_time = time.time()
    with torch.no_grad():
        output, _ = pose_model(image)
      # Get the end time.
    end_time = time.time()
    # Get the fps.
    fps = 1 / (end_time - start_time)
    # Add fps to total fps.
    total_fps += fps
    # Increment frame count.
    frame_count += 1

    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=pose_model.yaml['nc'], nkpt=pose_model.yaml['nkpt'], kpt_label=True)
    output = output_to_keypoint(output)

    #once we have the poses, we will be converting the image back again according to detector
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)  #changing again to locate the keypoints


    #calculating the IOU for ball and hoop
    bbox1 = roi.ball_coordinates(detect_frame)
    bbox2 = roi.hoop_coordinates(detect_frame)
    ball_hoop_iou = roi.calculate_iou(bbox1, bbox2)
    if not math.isnan(ball_hoop_iou):
      all_ball_hoop_ious.append(ball_hoop_iou)



    ############################ storing all of the frames numbers and respective ball coordinates  ########################################
    if frame_no not in frame_ball_cord:
       frame_ball_cord[frame_no] = []
    frame_ball_cord[frame_no].append(bbox1["x1"])
    frame_ball_cord[frame_no].append(bbox1["y1"])
    frame_ball_cord[frame_no].append(bbox1["x2"])
    frame_ball_cord[frame_no].append(bbox1["y2"])
    ##########################################################################################################################



    ############################getting all of the player_ids and their respective keypoints######################################3
    if frame_no not in frame_hand_keypoints:
       frame_hand_keypoints[frame_no] = {}
    
    if frame_no not in frame_feet_keypoints:
       frame_feet_keypoints[frame_no] = {}
    
    if frame_no not in frame_knee_keypoints:
       frame_knee_keypoints[frame_no] = {}

    id_key = get_keypoints(nimg, frame_width, frame_height)
    for id in id_key:
       for key in id_key[id]:
          if key ==9 or key == 10:
            if id not in frame_hand_keypoints[frame_no]:
              frame_hand_keypoints[frame_no][id] = {}

            if key not in frame_hand_keypoints[frame_no][id]:
              frame_hand_keypoints[frame_no][id][key] = []
            frame_hand_keypoints[frame_no][id][key] = id_key[id][key]   #storing all of the frames, ids and hands
    
    for id in id_key:
       for key in id_key[id]:
          if key ==15 or key == 16:
            if id not in frame_feet_keypoints[frame_no]:
              frame_feet_keypoints[frame_no][id] = {}

            if key not in frame_feet_keypoints[frame_no][id]:
              frame_feet_keypoints[frame_no][id][key] = []
            frame_feet_keypoints[frame_no][id][key] = id_key[id][key]   #storing all of the frames, ids and feet



    ################################### Main Visualisation ####################################################
    results = model.predict(source=detect_frame, conf=0.25)
    coordinates = results[0].boxes.xyxy
    classes = results[0].boxes.cls
    classes = torch.tensor(classes, dtype=torch.int32)
    
    for cord, cls in zip(coordinates, classes):
      x1 = int(cord[0])
      y1 = int(cord[1])
      x2 = int(cord[2])
      y2 = int(cord[3])
      cv2.rectangle(detect_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    # result.write(detect_frame)
    cv2.imshow('Frame',detect_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


  ################################### Main Visualisation ####################################################
   
    if frame_no not in frames:
        frames[frame_no] = np.zeros(detect_frame.shape)
    frames[frame_no] = detect_frame
    a = 8
    if len(all_ball_hoop_ious)>0:
      if max(all_ball_hoop_ious) >= ball_hoop_threshold:
        print(max(all_ball_hoop_ious))
        break

    frame_bank.append(frame_no)
    frame_no += 1

  
  else: 
    break
  
cap.release()
cv2.destroyAllWindows()


########################################################## Heuristic Logic #########################################################
decision_count = 0
decision = False
two_point = False

if len(all_ball_hoop_ious) != 0:
  if max(all_ball_hoop_ious) >= ball_hoop_threshold:   #if we have a certian balll-hoop threshold
      all_frames = frame_bank.copy()                
      for i in range(len(frame_bank)):
        
        if decision == True:
          break                 
        
        ref_frame = all_frames[-1]                    # moving back frame by frame to check in which frame the shot was made
        x1_b = frame_ball_cord[ref_frame][0]
        y1_b = frame_ball_cord[ref_frame][1]
        x2_b = frame_ball_cord[ref_frame][2]
        y2_b = frame_ball_cord[ref_frame][3]
        full_frame = frames[ref_frame]
        visframe = cv2.rectangle(full_frame, (x1_b, y1_b), (x2_b, y2_b), (255, 0, 0), 2)
        
     
        shot_taker = None
        for player_id in frame_hand_keypoints[ref_frame]:
          for key_point in frame_hand_keypoints[ref_frame][player_id]:
            hand = frame_hand_keypoints[ref_frame][player_id][key_point]
            cv2.circle(visframe, hand, 3, (255,0,0), -1)
            
          if x1_b-ball_hand_threshold < hand[0] and hand[0] < x2_b+ball_hand_threshold and y1_b-ball_hand_threshold < hand[1] and hand[1] < y2_b+ball_hand_threshold:
            if decision_count % 7 == 0:
              shot_taker = player_id

              for feet in frame_feet_keypoints[ref_frame][shot_taker]:
                foot = frame_feet_keypoints[ref_frame][shot_taker][feet]  #finding the place of the foot
                

                cv2.circle(visframe, foot, 3, (0,255,255), -1)
                # if decision_count % 5 == 0:
                region = shot_coordinate(full_frame, foot)    #we cant send the frame with bounding box for segmentation
                region = region.item()
                
                if region != 0:  # if the region detected is not in the background class
                  cv2.circle(visframe, (hand[0], hand[1]), 10, (255,0,0), -1)
                  cv2.circle(visframe, (foot[0], foot[1]), 10, (0,0,255), -1)


                  seg_map = get_seg_transform(full_frame)  #getting the segmentation map

                  #checking if there is only one player in the free throw circle
                  
                  if region == 2:  #if region is the free throw circle
                    all_frame_feet = []
                    for ev_player in frame_feet_keypoints[ref_frame]:
                      for shoe in frame_feet_keypoints[ref_frame][ev_player]:
                        shoes = frame_feet_keypoints[ref_frame][ev_player][shoe]
                        
                        if shoes[1] > 1088:
                          shoes[1] = 1087
                        if shoes[0] > 1920:
                          shoes[0] = 1919
                        
                        shoe_reg = seg_map[:, shoes[1], shoes[0]]
                        all_frame_feet.append(shoe_reg)
                    
                    free_feet = all_frame_feet.count(2)
                    if free_feet <= 2:
                      
                      font = cv2.FONT_HERSHEY_SIMPLEX
                      cv2.putText(visframe, 
                                  "Free Throw",
                                  (50, 50),
                                  font, 1,
                                  (0, 255, 255), 
                                  3,
                                  cv2.LINE_4)
                      
                      cv2.putText(visframe, 
                                  "shot taker", 
                                  (hand[0], hand[1]), 
                                  font, 1,
                                  (0, 255, 255), 
                                  2,
                                  cv2.LINE_4)
                      
                      # for i in range(50):
                        # result.write(visframe)
                      cv2.imshow('Frame',visframe)
                      cv2.waitKey(0)
                      decision = True   #for stoping the heuristic logic to go further

                    else:
                      two_point = True
                
              
                      

                  elif region == 2 and two_point == True:  #if multiple players were detected in the free throw circle
                      cv2.circle(visframe, (hand[0], hand[1]), 10, (255,0,0), -1)
                      cv2.circle(visframe, (foot[0], foot[1]), 10, (0,0,255), -1)

                      font = cv2.FONT_HERSHEY_SIMPLEX
                      cv2.putText(visframe,
                                  "two point region",
                                  (50, 50),
                                  font, 1,
                                  (0, 255, 255),
                                  3,
                                  cv2.LINE_4)
                      
                      cv2.putText(visframe, 
                                  "shot taker",
                                  (hand[0], hand[1]), 
                                  font, 1,
                                  (0, 255, 255), 
                                  2,
                                  cv2.LINE_4)
                      
                      # for i in range(50):
                      #   result.write(visframe)
                      cv2.imshow('Frame',visframe)
                      cv2.waitKey(0)
                      decision = True

                  else:  # if the player was not in the free throw circle then make a decision
                      
                      cv2.circle(visframe, (hand[0], hand[1]), 10, (255,0,0), -1)
                      cv2.circle(visframe, (foot[0], foot[1]), 10, (0,0,255), -1)

                      font = cv2.FONT_HERSHEY_SIMPLEX
                      cv2.putText(visframe,
                                  CLASSES[region],
                                  (50, 50),
                                  font, 1,
                                  (0, 255, 255), 
                                  3,
                                  cv2.LINE_4)
                      
                      cv2.putText(visframe, 
                                  "shot taker", 
                                  (hand[0], hand[1]), 
                                  font, 1,
                                  (0, 255, 255), 
                                  2,
                                  cv2.LINE_4)
                      
                  
                      # for i in range(50):
                      #   result.write(visframe)
                      cv2.imshow('Frame',visframe)
                      cv2.waitKey(0)
                      decision = True
                  

          else:
            cv2.imshow("Frame", visframe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
              decision = True

        
            
        
        all_frames.pop(-1)
        decision_count+=1

  cap.release()
  cv2.destroyAllWindows()