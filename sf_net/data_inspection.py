import json
import os
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import geopandas as gpd
import numpy as np
import shutil




# fold = os.listdir("/home/adlytic/Desktop/Osama/semantic-segmentation/output/test_masks")

# for file in fold:
#     print(np.unique(cv2.imread("/home/adlytic/Desktop/Osama/semantic-segmentation/output/test_masks/"+file)))

data3 = open("new3.json")
f3 = json.load(data3)
print(len(f3))
print(type(f3[0]))

data2 = open("new2.json")
f2 = json.load(data2)
print(len(f2))
print(type(f2[0]))

data1 = open("new.json")
f1 = json.load(data1)
print(len(f1))
print(type(f1[0]))

import requests

#f1
# print("first Json")
# for i in range(70): 
#     image_path = "new_images_2/image_" + str(i) + ".jpg"
#     mask_path = "new_annotations_2/image_" + str(i) + ".png"

#     print("image_" + str(i) + ".jpg")

#     with open(image_path, 'wb') as handle:
        
#         id = f1[i]["ID"]
#         #print("ID:", id)

#         img_url = f1[i]["Labeled Data"]
#         #print("img url:", img_url, "\n")

#         response = requests.get(img_url, stream=True)

#         if not response.ok:
#             print(response)

#         for block in response.iter_content(1024):
#             if not block:
#                 break

#             handle.write(block)

#         print("image saved!")
#         img_shape = cv2.imread(image_path).shape
#         road = np.zeros(img_shape)

#         soil = []
#         asphalt = []
#         concrete = []

#         print("--> making labels")

#         for labels in f1[i]["Label"]["objects"]:
#             label = labels["value"]
#             #print("label:", label)
#             #label_url = labels['instanceURI']
#             #print("label_url:", label_url)
            
#             polygon = []
#             for crr in labels["polygon"]:
#                 polygon.append((int(crr["x"]), int(crr["y"])))
            
#             if label == "soil":
#                 cv2.fillPoly(road, pts = [np.array(polygon)], color = [3,3,3])
#             elif label == "concrete":
#                 cv2.fillPoly(road, pts = [np.array(polygon)], color = [1,1,1])
#             elif label == "asphalt":
#                 cv2.fillPoly(road, pts = [np.array(polygon)], color = [2,2,2])
        
#         road = road[:,:,0]
#         background = np.where(road==0)
#         road[background] = 4

#         road = road.reshape((road.shape[0], road.shape[1], 1))
#         cv2.imwrite(mask_path, road)
#         print("mask saved!")
#         #cv2.waitKey(0)


#         # concrete = np.where(road!=0)
#         # road[concrete] = 1


#         print("-------------------------------")

print("\nSecond Json")
for i in range(97): 
    image_path = "new_images_2/image_" + str(i+70) + ".jpg"
    mask_path = "new_annotations_2/image_" + str(i+70) + ".png"

    print("image_" + str(i+70) + ".jpg")

    with open(image_path, 'wb') as handle:
        
        id = f2[i]["ID"]
        #print("ID:", id)

        img_url = f2[i]["Labeled Data"]
        #print("img url:", img_url, "\n")

        response = requests.get(img_url, stream=True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)

        print("image saved!")
        img_shape = cv2.imread(image_path).shape
        road = np.zeros(img_shape)

        soil = []
        asphalt = []
        concrete = []

        print("--> making labels")

        for labels in f2[i]["Label"]["objects"]:
            label = labels["value"]
            #print("label:", label)
            #label_url = labels['instanceURI']
            #print("label_url:", label_url)
            
            polygon = []
            for crr in labels["polygon"]:
                polygon.append((int(crr["x"]), int(crr["y"])))
            
            if label == "soil":
                cv2.fillPoly(road, pts = [np.array(polygon)], color = [3,3,3])
            elif label == "concrete":
                cv2.fillPoly(road, pts = [np.array(polygon)], color = [1,1,1])
            elif label == "asphalt":
                cv2.fillPoly(road, pts = [np.array(polygon)], color = [2,2,2])
        
        road = road[:,:,0]
        background = np.where(road==0)
        road[background] = 4

        road = road.reshape((road.shape[0], road.shape[1], 1))
        cv2.imwrite(mask_path, road)
        print("mask saved!")
        #cv2.waitKey(0)


        # concrete = np.where(road!=0)
        # road[concrete] = 1


        print("-------------------------------")


print("\nThird Json")
for i in range(150): 
    image_path = "new_images_2/image_" + str(i+167) + ".jpg"
    mask_path = "new_annotations_2/image_" + str(i+167) + ".png"

    print("image_" + str(i+167) + ".jpg")

    with open(image_path, 'wb') as handle:
        
        id = f3[i]["ID"]
        #print("ID:", id)

        img_url = f3[i]["Labeled Data"]
        #print("img url:", img_url, "\n")

        response = requests.get(img_url, stream=True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)

        print("image saved!")
        img_shape = cv2.imread(image_path).shape
        road = np.zeros(img_shape)

        soil = []
        asphalt = []
        concrete = []

        print("--> making labels")

        for labels in f3[i]["Label"]["objects"]:
            label = labels["value"]
            #print("label:", label)
            #label_url = labels['instanceURI']
            #print("label_url:", label_url)
            
            polygon = []
            for crr in labels["polygon"]:
                polygon.append((int(crr["x"]), int(crr["y"])))
            
            if label == "soil":
                cv2.fillPoly(road, pts = [np.array(polygon)], color = [3,3,3])
            elif label == "concrete":
                cv2.fillPoly(road, pts = [np.array(polygon)], color = [1,1,1])
            elif label == "asphalt":
                cv2.fillPoly(road, pts = [np.array(polygon)], color = [2,2,2])
        
        road = road[:,:,0]
        background = np.where(road==0)
        road[background] = 4

        road = road.reshape((road.shape[0], road.shape[1], 1))
        cv2.imwrite(mask_path, road)
        print("mask saved!")
        #cv2.waitKey(0)


        # concrete = np.where(road!=0)
        # road[concrete] = 1


        print("-------------------------------")    

# data3 = open("new3.json")
# f3 = json.load(data3)
# images = data['images']
# annotations = data['annotations']


# for img in images:
#     img_name = img['file_name']
#     print(img_name)
#     img_id = img['id']

#     annos = []
    
#     for i, anno in enumerate(annotations):
#         if anno['image_id'] == img_id:
#             annos.append(anno)

    
#     cate_1 = []
#     cate_2 = []
#     cate_3 = []

#     for draw in annos:
#         if draw['category_id'] == 1:
#             cate_1.append(draw['segmentation'][0])
#         elif draw['category_id'] == 2:
#             cate_2.append(draw['segmentation'][0])
#         elif draw['category_id'] == 3:
#             cate_3.append(draw['segmentation'][0])


#     path = "Temp_data/all_images/" + img_name
#     footage = cv2.imread(path)
#     #road = np.zeros((footage.shape[0], footage.shape[1], 3))

#     road = np.zeros((footage.shape[0], footage.shape[1]))

#     shapes = []

#     for label in cate_1: 
#         polygon = []
#         for i in range(0, len(label), 2):
#             polygon.append((int(label[i]), int(label[i+1])))
#         cv2.fillPoly(road, pts = [np.array(polygon)], color = (1))
#         # concrete = np.where(road!=0)
#         # road[concrete] = 1


#     for label in cate_2: 
#         polygon = []
#         for i in range(0, len(label), 2):
#             polygon.append((int(label[i]), int(label[i+1])))
#         cv2.fillPoly(road, pts = [np.array(polygon)], color = (2))
#         # asphalt = np.where((road!=0) & (road!=1))
#         # road[asphalt] = 2


#     for label in cate_3: 
#         polygon = []
#         for i in range(0, len(label), 2):
#             polygon.append((int(label[i]), int(label[i+1])))
#         cv2.fillPoly(road, pts = [np.array(polygon)], color = (3))
#         # soil = np.where((road!=0) & (road!=1) & (road!=2))
#         # road[soil] = 3

#     road[np.where(road==0)] = 4

#     road = road.reshape((road.shape[0], road.shape[1], 1))
    
#     cv2.imwrite("masks_new/" + img_name.split(".")[0] + ".png",road)
#     cv2.imshow("road", to_show)
#     cv2.waitKey(0)

#     for i in range(background.shape[0]):
#         for j in range(background.shape[1]):
#             if concrete[i,j] == 0 and asphalt[i,j] == 0 and soil[i,j] == 0:
#                 background[i,j] = 1
    
    
    
#     road = np.zeros((footage.shape[0], footage.shape[1], 4))
#     road[:,:,0] = concrete
#     road[:,:,1] = asphalt
#     road[:,:,2] = soil
#     road[:,:,3] = background


#     #print(road.shape, "\n")
#     cv2.imwrite("masks/" + img_name.split(".")[0] + ".png", road)



# folder = "masks"
# for i in os.listdir(folder):
#     shutil.copy("images/" + i, "images_with_labels/" + i)
    


#--------------------------------------------
# folder = "images/big_jpgs/"
# folder_val = "images/big_jpgs_val/"

# for i in os.listdir(folder):
#     path = folder + i
#     img = cv2.imread(path)
#     #print("annotations/training/" + i.split(".")[0] + ".png")
#     cv2.imwrite("images/training/" + i.split(".")[0] + ".jpg",img)


# train = "annotations/training/"
# val = "annotations/validation/"

#-----------------------------------
# for i in os.listdir(val):
#     path = val + i
#     img = cv2.imread(path)
#     print(img.dtype)
#     img = img.astype('float16')
#     #print(img.dtype)
#     cv2.imwrite(path, img)

#--------------------------------------


# folder = "/home/adlytic/Desktop/Osama/semantic-segmentation/data/road_segmentation/segmentation_2/annotations/validation/"
# for i in os.listdir(folder):
#     mask = cv2.imread(folder + i)
#     mask[np.where(mask==1)] = 50
#     mask[np.where(mask==2)] = 150
#     mask[np.where(mask==3)] = 250

# 
#     cv2.imwrite("for_muaz/" + i, mask)

#--------------------------------------------------

# folder = "Temp_data/all_images/"
# folder2 = "Temp_data/images_with_labels/"

# for i in os.listdir(folder):
#      if i.lower() not in os.listdir(folder2):
#          shutil.copy(folder + i, "Temp_data/unannotated_images/" + i)

# print(len(os.listdir(folder)), len(os.listdir(folder2)))