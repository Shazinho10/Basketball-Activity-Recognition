from main_2 import Shot_Classification
CLASSES = ["background", "3point_Region", "freethrow_circle", "2point_region"]

video_path = "testing_videos/three_point/three_13_c_2.mp4"
even_classification = Shot_Classification(video_path, visualise=True) #visualise=False frame display is not required.
even_classification.data_collection()
region = even_classification.heuristic_approach()

print(f"Shot Region: {CLASSES[region]}")