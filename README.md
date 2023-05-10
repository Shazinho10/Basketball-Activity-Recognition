This approach makes use of YOLOv8 as a detector, YOLOv7 Pose for landmark detection and Segformer for region segmentation. 
As soon as the ball reaches the hoop, the algorithm will track back and check the overlap of the ball and hand in the latest frame. Then the foot of the same player will be extracted  and then that region is sent to the segmentation map for finding the region of shot.

1. The weights for the detector should be placed in the “detection_weights” folder.
2. The segmentation model weights need to be placed at the following path:
“sf_net/output/segformer_results”.
3. The pose YOLOv7 pose weights should be in the main directory.
“pose_approach.py” contains the whole heuristic development.
4. For testing the videos Insert the path of the video in the variable “vid_path” variable in “pose_approach.py” file.
5. The “ball_hoop_threshold” is to be treated as a hyperparameter and it should be adjusted after testing on a large dataset.
6. To test the free throw approach run the "free_throw.py" file.




https://github.com/Shazinho10/Basketball-Activity-Recognition/assets/96534007/ed2b1ff6-a09d-459c-8ac6-29a56f2908a9



https://github.com/Shazinho10/Basketball-Activity-Recognition/assets/96534007/f07ac5aa-21e4-4f54-80af-74d8244209c8

