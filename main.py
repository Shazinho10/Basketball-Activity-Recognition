from ultralytics import YOLO
import cv2
import torch
from roi import ROI
import numpy as np
from utils.datasets import letterbox
from field_classification import shot_coordinate, get_seg_transform, yolo_shot_coordinate
# from field_classification import shot_classification
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint
import os
import time
from torchvision import transforms
from multipose_handler import get_keypoints
import math
from utilities import visualize_result
from yolo_segmentation import get_yolo_seg_map


#################################### Warmup of the Models ################


device = "cuda" if torch.cuda.is_available() else "cpu"

#YOLOV8 Detector
model = YOLO("detection_weights/best_12th_april.pt")
roi = ROI(model)

#YOLOv7 Multi Pose Estimator
pose_weigths = torch.load('yolov7-w6-pose.pt')
pose_model = pose_weigths['model']
pose_model = pose_model.half().to(device)
_ = pose_model.eval()

# CLASSES = ["background", "3point_Region", "freethrow_circle", "2point_region"]   #different regions of the basketball court 
CLASSES = ["background", "2point_region", "3point_Region", "freethrow_circle"]

ball_hoop_threshold = 0.35   #this is the hyper parameter that can be handled further
ball_hand_threshold = 10     #determines how close the ball is to the hand bounding box

class Shot_Classification():
  def __init__(self, vid_path, visualise=False):
    self.vid_path = vid_path
    # initializing all of the datastructures which will store respective information
    self.frame_no = 0
    self.all_ball_hoop_ious = []    #storing the iou for ball and hoop in every frame
    self.frame_max_iou = {}  #storing the frame number and the max IOU
    self.frame_ball_cord = {}
    self.frame_hand_keypoints = {} #storing all of the keypoints of hands against the frame number
    self.frame_feet_keypoints = {} #storing all of the keypoints of feet against the frame number
    self.frame_knee_keypoints = {} #storing the knee keypoints

    self.frames = {}
    self.frame_bank = []
    self.visualise = visualise
    self.cap = cv2.VideoCapture(vid_path)
    self.frame_width = int(self.cap.get(3))
    self.frame_height = int(self.cap.get(4))


  def data_collection(self):  #this method collects the information of players, ball and the hoop
    #testing the max IOU
    
    if (self.cap.isOpened()== False): 
      print("Error opening video stream or file")

    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.
    count = 0
    stop_frame = False

    while(self.cap.isOpened()):
      ret, frame = self.cap.read()
      if ret == True:
        orig_image = frame
        detect_frame = letterbox(frame, (self.frame_width), stride=64, auto=True)[0]

        #converting the frame according to the pose estimation model
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = letterbox(image, (self.frame_width), stride=64, auto=True)[0]
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

        player_boxes = roi.player_coordinates(detect_frame)
        ball_hoop_iou = roi.calculate_iou(bbox1, bbox2)
        
        if not math.isnan(ball_hoop_iou):
          self.all_ball_hoop_ious.append(ball_hoop_iou)



        ############################ storing all of the frames numbers and respective ball coordinates  ########################################
        if self.frame_no not in self.frame_ball_cord:
          self.frame_ball_cord[self.frame_no] = []
        self.frame_ball_cord[self.frame_no].append(bbox1["x1"])
        self.frame_ball_cord[self.frame_no].append(bbox1["y1"])
        self.frame_ball_cord[self.frame_no].append(bbox1["x2"])
        self.frame_ball_cord[self.frame_no].append(bbox1["y2"])


        ############################# storing all of the frame numbers and respective player coordinates #################################################
        players = roi.player_coordinates(detect_frame) #dictionary with player number and coordinates

        ############################getting all of the player_ids and their respective keypoints######################################3
        if self.frame_no not in self.frame_hand_keypoints:
          self.frame_hand_keypoints[self.frame_no] = {}
        
        if self.frame_no not in self.frame_feet_keypoints:
          self.frame_feet_keypoints[self.frame_no] = {}
        
        if self.frame_no not in self.frame_knee_keypoints:
          self.frame_knee_keypoints[self.frame_no] = {}

        id_key = get_keypoints(nimg, self.frame_width, self.frame_height)
        for id in id_key:
          for key in id_key[id]:
              if key ==9 or key == 10:
                if id not in self.frame_hand_keypoints[self.frame_no]:
                  self.frame_hand_keypoints[self.frame_no][id] = {}

                if key not in self.frame_hand_keypoints[self.frame_no][id]:
                  self.frame_hand_keypoints[self.frame_no][id][key] = []
                self.frame_hand_keypoints[self.frame_no][id][key] = id_key[id][key]   #storing all of the frames, ids and hands
        
        for id in id_key:
          for key in id_key[id]:
              if key ==15 or key == 16:
                if id not in self.frame_feet_keypoints[self.frame_no]:
                  self.frame_feet_keypoints[self.frame_no][id] = {}

                if key not in self.frame_feet_keypoints[self.frame_no][id]:
                  self.frame_feet_keypoints[self.frame_no][id][key] = []
                self.frame_feet_keypoints[self.frame_no][id][key] = id_key[id][key]   #storing all of the frames, ids and feet



        ################################### Main Visualisation
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
        
        if self.visualise:
          cv2.imshow('Frame',detect_frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break


      ################################### Main Visualisation ####################################################
        
        if self.frame_no not in self.frames:
            self.frames[self.frame_no] = np.zeros(detect_frame.shape)
        self.frames[self.frame_no] = orig_image

        if len(self.all_ball_hoop_ious)>0:
          if max(self.all_ball_hoop_ious) >= ball_hoop_threshold:
            
            bbox1_x2 = bbox1['x1'] + (bbox1['x2']//3)
            bbox1_x1 = bbox1['x1'] - (bbox1['x2']//2)
            # bbox1_x2 = bbox1_x2 // 2
            bbox1_y2 = bbox1['y1'] + bbox1['y2']
            # bbox1_y2 = bbox1_y2 // 2

            bbox1_y1 = bbox1['y1'] - (bbox1['y2']//2)
            bbox2_x2 = bbox2['x1'] + (bbox2['x2']//3)
            bbox2_x1 = bbox2['x1'] - (bbox2['x2']//2)
            # bbox2_x2 = bbox2_x2 // 2
            bbox2_y2 = bbox2['y1'] + bbox2['y2']
            bbox2_y1 = bbox2['y1'] - (bbox2['y2']//2)
            # bbox2_y2 = bbox2_y2 // 2

            cv2.rectangle(detect_frame, (bbox1['x1'], bbox1_y1), (bbox1_x2, bbox1_y2), (255,211,67), 5)
            cv2.rectangle(detect_frame, (bbox2['x1'], bbox2_y1), (bbox2_x2, bbox2_y2), (0, 0, 255), 5)
            # cv2.imshow("IOU FRAME", detect_frame)
            # cv2.waitKey(0)
            break

        self.frame_bank.append(self.frame_no)
        self.frame_no += 1
        

      
      else: 
        break
      
    self.cap.release()
    cv2.destroyAllWindows()


########################################################## HEURISTIC LOGIC #########################################################


  def heuristic_approach(self):  #the begining of the heuristic approach
    self.data_collection()
    decision_count = 0
    decision = False
    two_point = False
    track_back_storing = []
    region = None
    # both_feet_region = []
    # both_feet_cords = []

    if len(self.all_ball_hoop_ious) != 0:
      if max(self.all_ball_hoop_ious) >= ball_hoop_threshold:   #if we have a certian balll-hoop threshold
          all_frames = self.frame_bank.copy()                
          for i in range(len(self.frame_bank)):
            
            if decision == True:
              break                 
            
            ref_frame = all_frames[-1]                    # moving back frame by frame to check in which frame the shot was made
            track_back_storing.append(self.frames[ref_frame])
            x1_b = self.frame_ball_cord[ref_frame][0]
            y1_b = self.frame_ball_cord[ref_frame][1]
            x2_b = self.frame_ball_cord[ref_frame][2]
            y2_b = self.frame_ball_cord[ref_frame][3]
            full_frame = self.frames[ref_frame]
            visframe = cv2.rectangle(full_frame, (x1_b, y1_b), (x2_b, y2_b), (255, 0, 0), 2)
            
        
            shot_taker = None
            for player_id in self.frame_hand_keypoints[ref_frame]:
              for key_point in self.frame_hand_keypoints[ref_frame][player_id]:
                hand = self.frame_hand_keypoints[ref_frame][player_id][key_point]
                cv2.circle(visframe, hand, 3, (255,0,0), -1)
                
              if x1_b-ball_hand_threshold < hand[0] and hand[0] < x2_b+ball_hand_threshold and y1_b-ball_hand_threshold < hand[1] and hand[1] < y2_b+ball_hand_threshold:
                if decision_count % 2 == 0:
                  shot_taker = player_id

                  for feet in self.frame_feet_keypoints[ref_frame][shot_taker]:
                    foot = self.frame_feet_keypoints[ref_frame][shot_taker][feet]  #finding the place of the foot
                    ftx = foot[0]
                    fty = foot[1]

                    # both_feet_cords.append(foot)

                    cv2.circle(visframe, foot, 3, (0,255,255), -1)
                    region = shot_coordinate(full_frame, foot)    #we cant send the frame with bounding box for segmentation
                    # region = yolo_shot_coordinate(full_frame, foot)
                    region = region.item()
                    region = int(region)
                    # both_feet_region.append(region)

                    print("_"*20)
                    print(f"Initially region is {CLASSES[region]}")  
                    print("_"*20)


                    if region != 0 and region is not None:
                      seg_map = get_seg_transform(full_frame)  #getting the segmentation map
                      
                      #checking if there is only one player in the free throw circle
                      if region == 2 and two_point == False:
                        
                        # compensation of the poor segmentation of free throw circle
                        #for the case of free throw circle, only consider the region, behind the shot_takers hand
                        #this will compensate for the poor performance of the model in case of the semi circle
                        # if hand[0] > ftx:
                        #   seg_map = seg_map[:, :, :hand[0]]
                      
                        # elif hand[0] < ftx:
                        #   seg_map = seg_map[:, :, hand[0]:]
                        
                        all_frame_feet = []
                        for ev_player in self.frame_feet_keypoints[ref_frame]:
                          for shoe in self.frame_feet_keypoints[ref_frame][ev_player]:
                            shoes = self.frame_feet_keypoints[ref_frame][ev_player][shoe]
                            
                            if shoes[1] > 1088:
                              shoes[1] = 1087
                            if shoes[0] > 1920:
                              shoes[0] = 1919

                            if shoes[1] < seg_map.shape[1] and shoes[0] < seg_map.shape[2]:
                              shoe_reg = seg_map[:, shoes[1], shoes[0]]
                              all_frame_feet.append(shoe_reg)
                        
                        free_feet = all_frame_feet.count(2)
                        if free_feet <= 2:
                          if self.visualise:
                            visualize_result(visframe, hand=hand, foot=foot, region=region, CLASSES=CLASSES)
                          decision = True
                          print("_"*20)
                          print(f"region is {CLASSES[region]}")  
                          print("_"*20)
                          return region
                        
                        else:
                          two_point = True
                    
                  
                          
                      # if a two point shot has been made from the two point region
                      elif region == 2 and two_point == True:
                          # for ft in both_feet_cords:  #check each foot in this region
                          #   foot = ft

                          if self.visualise:
                            visualize_result(visframe, hand=hand, foot=foot, region=region, CLASSES=CLASSES, put_text="Two_Point_Region")
                          region = 3
                          print("_"*20)
                          print(f"region is {CLASSES[region]}")  
                          print("_"*20)
                          decision = True
                          return region

                      elif region == 3:  #compensation of the three point error
                        # for ft in both_feet_cords:  #check each foot in this region
                        #   foot = ft
                          # ftx = foot[0]
                          # fty = foot[1]

                          try:
                            future_image = track_back_storing[7]
                            players = roi.player_coordinates(future_image)
                            for plr in players:  #coordinates of all of the players
                              x1f = players[plr][0]
                              y1f = players[plr][1]
                              x2f = players[plr][2]
                              y2f = players[plr][3]
                              # future_image = cv2.rectangle(visframe, (x1f,y1f), (x2f,y2f), (255,0,0), 3))

                              if x1f < ftx and ftx < x2f and y1f < fty and fty < y2f:
                                bottom_center = ((x1f+x2f)//2, y2f)
                                region = region = shot_coordinate(future_image, bottom_center)
                                region = region.item()
                                # cv2.imshow("future_image", future_image)
                                # cv2.waitKey(0)

                                #calculating on the future frame and putting text onto the past one for better visualization
                                # cv2.circle(visframe, bottom_center, 5, (255, 255, 100), 5)
                                if self.visualise:
                                  visualize_result(visframe, hand=hand, foot=foot, region=region, CLASSES=CLASSES)
                                decision = True
                                print("_"*20)
                                print(f"region is {CLASSES[region]}")  
                                print("_"*20)
                                return region
                                break
                                

                              else:
                                if self.visualise:
                                  visualize_result(visframe, hand=hand, foot=foot, region=region, CLASSES=CLASSES)
                                decision = True
                                print("_"*20)
                                print(f"region is {CLASSES[region]}")  
                                print("_"*20)
                                return region

                          except Exception as e:  #For future frame where the number of frames might not be high, such as dunk shots
                            print(e)
                            if self.visualise:
                              visualize_result(visframe, hand=hand, foot=foot, region=region, CLASSES=CLASSES)
                            decision = True
                            print("_"*20)
                            print(f"region is {CLASSES[region]}")  
                            print("_"*20)
                            return region
                        

                      else: #if the player was in the three point region
                          # for ft in both_feet_cords:  #check each foot in this region
                          #   foot = ft
                          if self.visualise:
                            visualize_result(visframe, hand=hand, foot=foot, region=region, CLASSES=CLASSES)
                          decision = True
                          print("_"*20)
                          print(f"region is {CLASSES[region]}")  
                          print("_"*20)
                          return region
              else:
                if self.visualise:
                  cv2.imshow("Frame", visframe)
                  if cv2.waitKey(1) & 0xFF == ord('q'):
                    decision = True
                else:
                  pass

            
                
            
            all_frames.pop(-1)
            decision_count+=1

      self.cap.release()
      cv2.destroyAllWindows()


if __name__ == "__main__":
  video_path = "testing_videos/one_point/one_point_video_3_24_c_1.mp4"
  event_classification = Shot_Classification(video_path, visualise=True)
  # even_classification.data_collection()
  region = event_classification.heuristic_approach()