import cv2
import numpy as np
import math
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('--image_folder', type=str,
                    help='A required string argument')

parser.add_argument('--save_dir', type=str,
                    help='A required string argument')

parser.add_argument('--type', type=str,
                    help='annotations or images')

parser.add_argument('--slice_size', type=int,
                    help='slice dimensions')

args = parser.parse_args()

def checkRatio(x, y):

    #print("In check ratio method")
    
    #x = int(round(x / (math.gcd(50, 10)), 0))
    #y = int(round(y / (math.gcd(50, 10)), 0))

    '''
    if x == 0:
        x = 1
    if y == 0:
        y = 1
    '''

    bigger_dim = None

    #print(x, y)

    if x > y:
        x = round(x/y)
        y = 1
        bigger_dim = x

    else:
        #print("In else condition")
        y = round(y/x)
        x = 1
        bigger_dim = y

    #print(x, y, bigger_dim)

    return (x, y, bigger_dim)



# Generate slices using dynamic slicing algo
def slice_image(image, image_name, windowSize=1024):
   # print("in slice image method")
    h, w, c = image.shape
    #print(h, w)

    checkRatio(w, h)

    # if w < 1500 and h < 1500:
    #     print("perform inference on whole crop")
    #     #output_dict = run(model, image, conf_thres=0.45)
    #     #return output_dict

    widthIter = 1
    heightIter = 1
    maxHeightIterator=1

    maxHeightAchived = False
    maxWidthAchived = False
    iter = 0

    offset_dict = {}

    if heightIter * windowSize > h:
        #print("In if condition height less than 1024")
        while(widthIter * windowSize < w):
            iter += 1
            if ((heightIter + 0.5) * windowSize > h):
                #print("case 1 called")
                maxHeightAchived = True

                # print("width iter: ", widthIter)
                # print("Window Size: ", windowSize)
                # print((widthIter - 1 * windowSize))
                # print((widthIter) * windowSize)

                imageSlice = image[0: h, (widthIter - 1) * windowSize: (widthIter) * windowSize]
                #print(imageSlice.shape)

            slice_coordinates = {'x': (widthIter-1) * windowSize, 'y': (heightIter-1) * windowSize,
                                    'image': imageSlice}

            offset_dict[iter] = slice_coordinates

            widthIter += 1

    elif widthIter * windowSize > w:
        #print("In if condition width less than 1024")
        while(heightIter * windowSize < h):
            iter += 1
            if ((widthIter + 0.5) * windowSize > w):
                #print("case 1 called")
                maxWidthAchived = True

                imageSlice = image[(heightIter-1) * windowSize: (heightIter) * windowSize, 0: w]
                #print(imageSlice.shape)

            slice_coordinates = {'x': (widthIter-1) * windowSize, 'y': (heightIter-1) * windowSize,
                                    'image': imageSlice}

            offset_dict[iter] = slice_coordinates

            heightIter += 1

    else:
        while(widthIter * windowSize < w):
            #print("In first while loop")

            while (heightIter * windowSize < h):
                iter += 1

                if ((heightIter + 0.5) * windowSize > h) and ((widthIter + 0.5) * windowSize > w):
                    #print("case 1 called")
                    maxHeightAchived = True
                    maxWidthAchived = True

                    imageSlice = image[(heightIter-1) * windowSize: h, (widthIter-1) * windowSize: w]

                elif (widthIter + 0.5) * windowSize > w:
                    #print("case 2 called")

                    maxWidthAchived = True

                    imageSlice = image[(heightIter-1) * windowSize: (heightIter) * windowSize,
                                    (widthIter-1) * windowSize: w]

                elif (heightIter + 0.5) * windowSize > h:
                    #print("case 3 called")

                    maxHeightAchived = True

                    imageSlice = image[(heightIter-1) * windowSize: h,
                                    (widthIter-1) * windowSize: (widthIter) * windowSize]

                if maxWidthAchived == False and maxHeightAchived == False:
                    #print("case 4 called")
                    imageSlice = image[(heightIter-1) * windowSize: (heightIter) * windowSize,
                                    (widthIter-1) * windowSize: (widthIter) * windowSize]

                elif maxWidthAchived == False and maxHeightAchived == True and ((heightIter + 0.5) * windowSize < h):
                    #print("case 5 called")

                    imageSlice = image[(heightIter-1) * windowSize: (heightIter) * windowSize,
                                    (widthIter-1) * windowSize: (widthIter) * windowSize]

                elif maxWidthAchived == True and maxHeightAchived == False and ((widthIter + 0.5) * windowSize < w):
                    #print("case 5 called")

                    imageSlice = image[(heightIter-1) * windowSize: (heightIter) * windowSize,
                                    (widthIter-1) * windowSize: (widthIter) * windowSize]

                slice_coordinates = {'x': (widthIter-1) * windowSize, 'y': (heightIter-1) * windowSize,
                                    'image': imageSlice}

                offset_dict[iter] = slice_coordinates

                heightIter += 1

            maxHeightIterator= heightIter
            heightIter = 1
            widthIter += 1       

    limitWidth = (widthIter-1) * windowSize
    # limitHeight = (heightIter-1) * windowSize
    #print("maxHeightIterator: ", maxHeightIterator)
    limitHeight = (maxHeightIterator-1) * windowSize

    if maxHeightAchived:
        #print("Max height achieved")
        limitHeight = h

    if maxWidthAchived:
        #print("Max width achieved")
        limitWidth = w

    # Code for extracting remaining slices
    if maxWidthAchived and maxHeightAchived:
        pass
        # return the offset with images

    # Max. height achieved
    elif maxHeightAchived:
        remaining_width = w - limitWidth

        x, y, bigger_dim = checkRatio(remaining_width, h)

        loop_size = 0
        
        loop_size = h / bigger_dim
        round_size = round(loop_size)

        for i in range(1, bigger_dim + 1):

            imageSlice = image[(i-1) * round_size: i * round_size,
                            limitWidth: w]

            slice_coordinates = {'x': limitWidth, 'y': (i-1) * round_size, 'image': imageSlice}
            offset_dict["maxh" + str(i)] = slice_coordinates

    # Max. width achieved
    elif maxWidthAchived:
        remaining_height = h - limitHeight

        x, y, bigger_dim = checkRatio(remaining_height, w)

        loop_size = 0
        
        loop_size = w / bigger_dim
        round_size = round(loop_size)

        for i in range(1, bigger_dim + 1):
            imageSlice = image[limitHeight: h,
                                (i-1) * round_size: i * round_size]

            slice_coordinates = {'x': (i-1) * round_size, 'y': limitHeight, 'image': imageSlice}
            offset_dict["maxw" + str(i)] = slice_coordinates

    # Both max. width and max. height achieved 
    else:
        remaining_width = w - limitWidth
        remaining_height = h - limitHeight

        x, y, bigger_dim = checkRatio(remaining_width, h)

        loop_size = 0

        if w > h:
            loop_size = w / bigger_dim
            round_size = round(loop_size)

            # Slice remaining width over limitHeight
            for i in range(1, bigger_dim + 1):

                imageSlice = image[limitHeight: h,
                                (i-1) * round_size: i * round_size]

                slice_coordinates = {'x': (i-1) * round_size, 'y': limitHeight, 'image': imageSlice}
                offset_dict["maxwhw" + str(i)] = slice_coordinates

            if limitHeight > 0:
                a, b, bd = checkRatio(remaining_width, limitHeight)
                loop_size = limitHeight / bd
                round_size = round(loop_size)

                # Slice remaining height over limitWidth
                for i in range(1, bd + 1):

                    imageSlice = image[(i-1) * round_size: i * round_size,
                                limitWidth: w]

                    slice_coordinates = {'x': limitWidth, 'y': (i-1) * round_size, 'image': imageSlice}
                    offset_dict["maxwhh" + str(i)] = slice_coordinates

        elif h > w:
            loop_size = h / bigger_dim
            round_size = round(loop_size)

            # Slice remaining height over limitWidth
            for i in range(1, bigger_dim + 1):
                imageSlice = image[(i-1) * round_size: i * round_size,
                            limitWidth: w]

                slice_coordinates = {'x': limitWidth, 'y': (i-1) * round_size, 'image': imageSlice}
                offset_dict["maxwhh" + str(i)] = slice_coordinates

            a, b, bd = checkRatio(remaining_height, limitWidth)
            loop_size = limitWidth / bd
            round_size = round(loop_size)

            # Slice remaining width over limitHeight
            for i in range(1, bd + 1):

                image[limitHeight: h,
                                (i-1) * round_size: i * round_size]

                slice_coordinates = {'x': (i-1) * round_size, 'y': limitHeight, 'image': imageSlice}
                offset_dict["maxwhw" + str(i)] = slice_coordinates

    #print(offset_dict)

    #print("Length of offset dict: ", len(offset_dict))

    # slice_write_path = os.getcwd() + '/data/slicing_test/new_data_to_slice/new_data/sliced_original/'
    #slice_write_path = os.getcwd() + '/output/unannotated_slices/'
    slice_write_path = os.getcwd() + args.save_dir 
    #slice_write_path = os.getcwd() + '/data/slicing_test/new_data_to_slice/new_data/sliced_binary/'

    for key, value in offset_dict.items():
        #print("-------------")
        #print(key)
        #print(value)

        slice_x = value.get('x')
        slice_y = value.get('y')
        #print(slice_x, slice_y)


        slice_img = value.get('image')
        # print(slice_img)
        # cv2.imwrite("./data/slicing_test/slices/" + image_name + '_' + str(key) + '.png', slice_img[:,:,-1])
        # slice_img = cv2.resize(slice_img, (512, 512))
        # import pdb
        # pdb.set_trace()
        if args.type == 'images':
            cv2.imwrite(slice_write_path + image_name + '_' + str(key) + '.jpg', slice_img)
        elif args.type == 'annotations':
            cv2.imwrite(slice_write_path + image_name + '_' + str(key) + '.png', slice_img[:,:,-1])

        # [:,:,-1])

    '''
    updatedCords = {0: [], 1: [], 2: []}
    final_dict = {0: [], 1: [], 2: []}

    for key, value in offset_dict.items():
        print("-------------")
        print(key)
        print(value)

        slice_x = value.get('x')
        slice_y = value.get('y')
        print(slice_x, slice_y)

        slice_img = value.get('image')
        cv2.imwrite("./slice_images/" + str(key) + '.jpg', slice_img)

        output_dict = run(model, slice_img, conf_thres=0.45)
        print(output_dict)

        for key, value in output_dict.items():
            for i in range(len(output_dict[key])):
                output_dict[key][i]['x'] = output_dict[key][i]['x'] + int(slice_x)
                output_dict[key][i]['y'] = output_dict[key][i]['y'] + int(slice_y)
                updatedCords[key].append(output_dict[key][i])

    for key, value in updatedCords.items():
        print(key, value)

        for item in value:
            x = item['x']
            y = item['y']

            #x = int((x / 3072) * w)
            #y = int((y / 3072) * h)

            final_dict[key].append({'x': x, 'y': y})

            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    

    return [final_dict[0], final_dict[1] + final_dict[2]], image
    '''

    return offset_dict
      
if __name__ == '__main__':

    # img_dir = os.getcwd() + '/data/slicing_test/test7/'

    # img_dir = os.getcwd() + '/data/slicing_test/new_data_to_slice/new_data/orig-images/'
    img_dir = os.getcwd() + args.image_folder
    print(img_dir)
    #img_dir = os.getcwd() + '/data/slicing_test/new_data_to_slice/new_data/binary/'

    for img in sorted(os.listdir(img_dir)):
        print(img)
        if '.txt' not in img:
            img_name = img.split('.')[0]
            img_path = img_dir + img
            print("-->",img_path)
            image = cv2.imread(img_path)

            print("-----------------------------")
            print("Slicing image: ", img_name)
            print("-----------------------------")

            # offset_dict = split(image)
            size = args.slice_size
            offset_dict = slice_image(image, img_name, windowSize=size)
            
            #print(offset_dict)

    
        
