import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import time
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7-w6-pose.pt')
model = weigths['model']
model = model.half().to(device)
_ = model.eval()


def get_keypoints(frame, frame_width, frame_height):   #returns a dictionary with player id and their hand keypoints
    id_key = {}  #this will store the ids and the keypoints of the detections

    
    orig_image = frame
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = letterbox(image, (frame_width), stride=64, auto=True)[0]   #resizing the image for the inference of the frame
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    image = image.half()

    # Get the start time.
    start_time = time.time()
    with torch.no_grad():
        output, _ = model(image) 

    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)  #changing back so that it can be displayed again


    for idx in range(output.shape[0]):   #idx repressents eachy player
        kpts = output[idx, 7:].T
        steps = 3
        num_kpts = len(kpts) // steps

        if idx not in id_key:
            id_key[idx] = {}

        for keypoint in range(num_kpts):
            if keypoint == 9 or keypoint == 10 or keypoint==15 or keypoint == 16 or keypoint==13 or keypoint==14:
                if keypoint not in id_key[idx]:
                    id_key[idx][keypoint] = []
                x_coord, y_coord = kpts[steps * keypoint], kpts[steps * keypoint + 1]
                if keypoint==13 or keypoint==14:
                    y_coord = y_coord + 70   #if the foot is not detected, we go 50 steps below the knee
                # x_coord, y_coord = kpts[steps * keypoint], kpts[steps * keypoint]
                id_key[idx][keypoint].append(int(x_coord))
                id_key[idx][keypoint].append(int(y_coord))

            if steps == 3:
                conf = kpts[steps * keypoint + 2]
                if conf < 0.5:
                    continue

                
        
    return id_key



if __name__ == "__main__":
    import cv2
    import numpy as np
    video_path = 'testing/test_videos/two_48_c_2.mp4'

    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # storing all of the key points against the player in a frame
            player_hands = get_keypoints(frame, frame_width=frame_width, 
                                              frame_height=frame_height)

            a = 67
            cv2.imshow('Frame',frame)     
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break
        
        # When everything done, release the video capture object
    cap.release()
        
        # Closes all the frames
    cv2.destroyAllWindows()