import numpy as np
import matplotlib.pyplot as plt
import cv2
from sf_net.inference_for_handler import semseg
import torchvision.transforms as transforms
from torchvision import io
import torch
import yaml



with open(r'segformer_custom.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

model = semseg(cfg)
def get_seg_map(image):
    model = semseg(cfg)

    #because torch takes RGB images we have to convert them from cv2 format which takes BGR images.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))     # this is how torchvision and our model read the images  (C, H, W)
    image = torch.Tensor(image)
    image = torch.tensor(image, dtype=torch.uint8)   #the tesnors from cv2 are of float thus we need to convert them
    img = model.preprocess(image)
    seg_map = model.model_forward(img)

    #this is out segmentation map of the frame.
    seg_map = model.postprocess(img, seg_map, overlay=True)
    return seg_map

if  __name__ == "__main__":
    # image = cv2.imread("sf_net/data/basketball_train_data/images/validation/v4_00022200.jpg")
    # segmap = get_seg_map(image)
    # print(torch.unique(segmap))
    # print(segmap.shape)

    import cv2
    import numpy as np
    
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('testing/two_point/two_32_c_2.mp4')
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            segmap = get_seg_map(frame)
            segmap = segmap.detach().numpy()
            segmap = np.reshape(segmap, (512, 928, 1))
            # Display the resulting frame
            cv2.imshow('Frame',segmap)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
        