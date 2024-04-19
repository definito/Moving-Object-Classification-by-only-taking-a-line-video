# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:14:36 2022

@author: OCT
"""

# Reference Page:
# https://www.geeksforgeeks.org/python-play-a-video-using-opencv/

# importing libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os
# from os import listdir

###############################################################################
def extract_the_line(img, point_1, point_2):
    #-- Extract pixel value of a line btw any given two points on an image (img)...
    # Make a line with "num" points...
    # point_i = [x_i, y_i], x is column # in _pixel_ coordinates!!
    
    # Reference:
    # https://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array
    
    x1, y1 = point_1 
    x2, y2 = point_2
    length = int(np.hypot(x2-x1, y2-y1)) + 1 # distance btw two points, in order to avoid the situation where distance < 1 pixel; 
    x, y = np.linspace(x1, x2, length), np.linspace(y1, y2, length)
    
    # Extract the values along the line
    # Take the value from nearset neighbor
    zi_NN = img[y.astype(int), x.astype(int)]
    
    return zi_NN



def find_object_boundary(obj_contour_img, axis=1):
    # return countour's upper and lower boundary
    if axis == 0:
        d_x = np.average(obj_contour_img, axis=0)
        d_x[d_x < 1.2] = 0
        d_x[d_x > 0.9] = 10
        # plt.figure()
        # plt.plot(d_x)
        # plt.title('x-cut')
        return d_x
    elif axis == 1:
        d_y = np.average(obj_contour_img, axis=1)
        d_y[d_y < 1.2] = 0
        d_y[d_y > 0.9] = 10
        # plt.figure()
        # plt.plot(d_y)
        # plt.title('y-cut')
        return d_y
    else:
        return False
    
    
# Image parameter... To change for every video size ~depends
def object_cropper(img_section):
    obj_contour_img = np.zeros([img_section.shape[0], img_section.shape[1]])
    y0 = img_section[:, 0, 0]
    
    i = 0
    #Boundary detection loop for multiple object or single too
    while i < img_section.shape[1]:
        y1 = img_section[:, i, 0]
        mid = abs(np.vectorize(float)(y1) - np.vectorize(float)(y0))
        mid = np.convolve(mid, np.ones(100)/100, mode='same') # running average, here widow size is 30;
        mid[mid <= 10] = 0 #Brightness thursholds... 
        mid[mid > 1] = 10
        obj_contour_img[:,i] = mid[0:len(obj_contour_img[:,i])]
        del mid 
        i += 1
    #The boundrarry detection ends here
    
    # plt.figure()
    # plt.imshow(obj_contour_img)
    
    # Finding object boundrary here..... 
    d_x = find_object_boundary(obj_contour_img, axis=0)
    x_count = sum(i > 0 for i in np.diff(d_x))
    x_idx_left = np.argwhere(np.diff(d_x) > 0)
    x_idx_right = np.argwhere(np.diff(d_x) < 0)
    
    d_y = find_object_boundary(obj_contour_img, axis=1)
    y_count = sum(i > 0 for i in np.diff(d_y))
    # remove noise objects in y_direction 
    y_idx_top = np.argwhere(np.diff(d_y) > 0)
    y_idx_bottom = np.argwhere(np.diff(d_y) < 0)
    
    i = 0
    #Noise part- Removing noise from same image thresholds into a limit to 10                                
    while i < len(y_idx_bottom):
        if len(y_idx_top)==0 and len(y_idx_bottom)==0 and (y_idx_bottom[i] - y_idx_top[i]) < 10:
            d_y[int(y_idx_top[i]-1):int(y_idx_bottom[i]+1)] = 0
            y_count -= 1
            y_idx_top = np.delete(y_idx_top, i)
            y_idx_bottom = np.delete(y_idx_bottom, i)
            i -= 1
            
        i += 1 
    
    # plt.figure()
    # plt.plot(dy_y)
    # plt.plot(dy_x)
    # plt.title(f'The {cropped_imag_num}th detection boundary.')
    
    if (x_count==1 or x_count==0) and y_count==1:
        global cropped_imag_num
        # when object is on the image's margin
        if len(y_idx_top)==0:
            y_idx_top = np.array([int(0)])
        if len(y_idx_bottom)==0:
            y_idx_bottom = np.array([img_section.shape[0]])
            
        cropped_img = img_section[int(max(0, y_idx_top[0]-30)):int(min(y_idx_bottom[0]+10, img_section.shape[0])),:,:]
        print(f"shape of {cropped_imag_num}th cropped image: {cropped_img.shape}")
        if cropped_img.shape[0]-(30+10) > 20:
            # plt.figure()
            # plt.imshow(cropped_img)
            
            # make a folder by the name given below, if not already built before
            foldername = "Test folder PNG"
            if not os.path.isdir(foldername):
                os.mkdir(foldername)
             
            # Or save cropped image to a folder
            mpimg.imsave(foldername+f'\object_{cropped_imag_num}.png', cropped_img/255)
            cropped_imag_num += 1 # count cropped image number
        else:
            0==0 # do nothing, save nothing
            
    elif x_count >= y_count and x_count > 1:
        cropped_img_x_1 = img_section[:,0:int((x_idx_right[0]+x_idx_left[1])/2),:]
        object_cropper(cropped_img_x_1)
        cropped_img_x_2 = img_section[:,int((x_idx_right[0]+x_idx_left[1])/2):,:]
        object_cropper(cropped_img_x_2)
    elif y_count > x_count and y_count > 1:
        cropped_img_y_1 = img_section[0:int((y_idx_bottom[0]+y_idx_top[1])/2),:,:]
        object_cropper(cropped_img_y_1)
        cropped_img_y_2 = img_section[int((y_idx_bottom[0]+y_idx_top[1])/2):,:,:]
        object_cropper(cropped_img_y_2)
    else:
        0==0 # do nothing 
        


def fft_filer_lines(grey_img):
    grey_img_fft = np.fft.fftshift(np.fft.fft2(grey_img))
    
    f_hor = 20
    f_ver = 50
    my_mask = np.ones_like(grey_img)
    my_mask[:int(grey_img.shape[0]/2 - f_hor), int(grey_img.shape[1]/2 - f_ver):int(grey_img.shape[1]/2 + f_ver)] = 0
    my_mask[-int(grey_img.shape[0]/2 - f_hor):,int(grey_img.shape[1]/2 - f_ver):int(grey_img.shape[1]/2 + f_ver)] = 0
    
    # plt.figure(num=None, figsize=(8, 6), dpi=80)
    # plt.imshow(abs(my_mask), cmap='gray');
    
    new_graph = abs(np.fft.ifft2(grey_img_fft * my_mask))
    # plt.figure(num=None, figsize=(8, 6), dpi=80)
    # plt.imshow(new_graph, cmap='gray');
    
    return new_graph



###############################################################################



###############################################################################
# video_path = "Z:/78_anuerbahati/Private/GAUSS/My courses at Uni/Lecture_Modern_Image_Processing/project_ideas/Project videos/Complex view angle/"
# video_name = "In front of hospital entrance_Ayiben"
# video_format = ".mp4"

img_path = "D:/STUDY/Project/"
# video_name = "On the bridge from front _ Ayiben"
img_name = "2D_output_directly_from_video_Bridge_noon_1min"
img_format = ".png"
img_file = img_path + img_name + img_format 
bscan = mpimg.imread(img_file)
A = bscan + 0.0
if_vertical = True

# line_rgb = bscan
fig = plt.figure(figsize=(16, 50), dpi=50)
if if_vertical == True: # vertical cutting
    plt.imshow(bscan)
else: # horizontal cutting
    plt.imshow(cv2.rotate(bscan, cv2.ROTATE_90_COUNTERCLOCKWISE))
plt.title("lines from all frames", fontsize=20)
plt.xlabel("X pixel scvaling", fontsize=20)
plt.ylabel("frame number", fontsize=20)
plt.show()
###############################################################################



###############################################################################
# determine time-scale
my_img = bscan*255 + 0.0
line_aver_r = np.average(my_img[..., 0], axis=0)
line_aver_g = np.average(my_img[..., 1], axis=0)
line_aver_b = np.average(my_img[..., 2], axis=0)

plt.figure()
plt.plot(line_aver_r)
plt.plot(line_aver_g)
plt.plot(line_aver_b)
plt.title('average of all vertical lines from RGB channels')

diff_r = abs(np.diff(line_aver_r)) # absolute value of 1st order derivitive fora each channel
diff_g = abs(np.diff(line_aver_g))
diff_b = abs(np.diff(line_aver_b))

diff_max = np.maximum.reduce([diff_r, diff_g, diff_b])  # take max value elementwisely
diff_max[diff_max < 0.6] = 0
diff_max = np.convolve(diff_max, np.ones(10)/10, mode='same') # running average, here widow size is 10;
diff_max[diff_max > 0.1] = 10
diff_max[diff_max != 10] = 0
diff_max[0:2] = 0 
diff_max[-1] = 0 # make sure that 

plt.figure()
plt.plot(diff_r)
plt.plot(diff_g)
plt.plot(diff_b)
plt.plot(diff_max)
plt.title('1st order derivitive of RGB channels and object boundary over time')


# determine x-scale
rising_edge_detection = np.diff(diff_max)
# rising_edge_detection = np.convolve(rising_edge_detection, np.ones(2)/2, mode='same') # running average, here widow size is 2;
rising_edge_detection[abs(rising_edge_detection) < 10/2] = 0
vehicle_count = sum(i > 0 for i in rising_edge_detection)
left_idx = np.argwhere(rising_edge_detection > 0)
right_idx = np.argwhere(rising_edge_detection < 0)

plt.figure()
plt.plot(left_idx)
plt.plot(right_idx)

# fig = plt.figure(figsize=(16, 50), dpi=50)
# N = 7
# plt.imshow(my_img[:, int(left_idx[N]-10):int(right_idx[N]+10), :])
# # determine y-scale
# # N = 10#(1 object) # 77#(3 cars) # 12/15/40#(2 cars)
# # global cropped_imag_num
# # cropped_imag_num = 0
# test_img = bscan[:, int(left_idx[N]-10):int(right_idx[N]+10), :]
# plt.figure()
# plt.imshow(test_img)
# object_cropper(test_img)

global cropped_imag_num
cropped_imag_num = 1

i = 0
while i < right_idx.shape[0]:
    print(f'image = {i + 1}..')
    # test_img = line_rgb[:, int(left_idx[i]-10):int(right_idx[i]+10), :]
    test_img = my_img[:, max(0, int(left_idx[i]-10)):min(int(right_idx[i]+10), bscan.shape[1]), :]
    object_cropper(test_img)
    i += 1


###############################################################################






# # import the modules
# import os
# from os import listdir
 
# # get the path/directory
# folder_dir = "Z:/78_anuerbahati/Private/GAUSS/My courses at Uni/Lecture_Modern_Image_Processing/project_ideas/Codes/Ayiben's cropped pictures - side view/Bus"
# for images in os.listdir(folder_dir):
#     # check if the image ends with png
#     if (images.endswith(".png")):
#         print(images)
