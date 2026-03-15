
# Basketball Activity Recognition

This Project aims to classify different shot types in basketball matches from a broadcast view

---

## 🏀 Method Overview

This approach combines three state-of-the-art computer vision models:

* **YOLOv8** — Object detection (ball, hoop, players)
* **YOLOv7 Pose** — Player landmark estimation
* **SegFormer** — Court region segmentation

When the ball reaches the vicinity of the hoop, the algorithm performs a backward analysis to determine the shooting context. It checks the overlap between the ball and the player’s hand in the most recent frame to identify the shooter. The system then extracts the foot position of the same player and maps this region onto the segmentation output to determine the shot location on the court.

---

## ⚙️ Setup Instructions

1. **Detector Weights**
   Place the YOLOv8 detection weights inside the folder:

   ```
   detection_weights/
   ```

2. **Segmentation Weights**
   Place the SegFormer model weights at:

   ```
   sf_net/output/segformer_results/
   ```

3. **Pose Estimation Weights**
   Place the YOLOv7 Pose weights in the project’s root directory.

4. **Main Implementation File**
   The file `pose_approach.py` contains the complete heuristic pipeline.

5. **Video Testing**
   To test on a video, set the video path in the variable:

   ```
   vid_path
   ```

   inside `pose_approach.py`.

6. **Ball–Hoop Threshold (Hyperparameter)**
   The parameter `ball_hoop_threshold` controls the detection of ball proximity to the hoop.
   It should be tuned empirically using a large and diverse dataset.

7. **Free Throw Evaluation**
   To test the free-throw detection pipeline, run:

   ```
   free_throw.py
   ```

# Sample Results
Below are example outputs demonstrating the system in action:

https://github.com/Shazinho10/Basketball-Activity-Recognition/assets/96534007/ed2b1ff6-a09d-459c-8ac6-29a56f2908a9

https://github.com/Shazinho10/Basketball-Activity-Recognition/assets/96534007/f07ac5aa-21e4-4f54-80af-74d8244209c8
