# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:57:21 2022

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


def extract_the_line(img, point_1, point_2):
    #-- Extract pixel value of a line btw any given two points on an image (img)...
    # Make a line with "num" points...
    # point_i = [x_i, y_i], x is column number in _pixel_ coordinates!!
    
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

        


def frame_count(video_path, manual=False):
    # Reference: 
    # https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python
    def manual_count(handler):
        frames = 0
        while True:
            status, frame = handler.read()
            if not status:
                break
            frames += 1
        return frames 

    cap = cv2.VideoCapture(video_path)
    
    if manual: # Slow, inefficient but 100% accurate method 
        frames = manual_count(cap)
    
    else: # Fast, efficient but inaccurate method
        try:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            frames = manual_count(cap)
    cap.release()
    return frames



###############################################################################
# video_path = "Z:/78_anuerbahati/Private/GAUSS/My courses at Uni/Lecture_Modern_Image_Processing/project_ideas/Project videos/Direct view/"
# # video_name = "On the bridge from front _ Ayiben"
# video_name = "Thailand traffic footage for object recognition #4 _ downloaded from Youtube"
# # video_name = "Road traffic video for object recognition _ downloaded from Youtube" 
# video_format = ".mp4"

video_path = "D:/STUDY/Project/"
# video_name = "In front of hospital from side in the evening _ Ayiben"
# video_name = "In front of hospital entrance_Ayiben"
# video_name = "On the garage from side 01 _ Ayiben" 
# video_name = "On the garage from side 02 _ Ayiben" 
video_name = "Bridge_noon_1min" 
video_format = ".mp4"

video_file = video_path + video_name + video_format 

# Just play the video
cap = cv2.VideoCapture(video_file)
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")
 
# play until video is completed
while(cap.isOpened()): 
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
    # Display the resulting frame
        cv2.imshow('Cars on highway', frame)
         
    # Press lower case "q" on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
        
    # Break the loop
    else:
        break
 

# when everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
###############################################################################


###############################################################################
# show one frame and cutting location 
# p_1 = [300, 900] # [x_1, y_1]
# p_2 = [300, 1500] # "In front of hospital from side in the evening _ Ayiben"

# p_1 = [620, 10] # [x_1, y_1]
# p_2 = [620, 1200] # "In front of hospital entrance_Ayiben"

# p_1 = [20, 160] # [x_1, y_1]
# p_2 = [20, 1060] # "On the garage from side 01 _ Ayiben"

# p_1 = [20, 160] # [x_1, y_1]
# p_2 = [20, 1060] # "On the garage from side 02 _ Ayiben"

p_1 = [60, 0] # [x_1, y_1]
p_2 = [60, 700] # "On the garage from side 02 _ Ayiben"


length = int(np.hypot(p_2[0]-p_1[0], p_2[1]-p_1[1])) + 1
# # Are you cutting roughly vertically?
if_vertical = True
# if_vertical = False 

frame0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# color scheme: RGB, BGR, GRB, GRAY, CIE, YCrCb, HSV, HSL 
example_line = extract_the_line(frame0, p_1, p_2)

fig, axs = plt.subplots(2)
x = [p_1[0], p_2[0]]
y = [p_1[1], p_2[1]]
axs[0].plot(x, y)
axs[0].plot(x, y, color="white", linewidth=3)
axs[0].imshow(frame0)
axs[1].plot(example_line)
plt.title("Cutting", fontsize=20)
plt.xlabel("X pixel scaling", fontsize=20)
plt.ylabel("Y pixels scaling", fontsize=20)
plt.savefig(video_path+"Cutting_"+video_name+".png")
###############################################################################


###############################################################################
# Extract all lines to one matrix 
frame_start = 1
frame_until = frame_count(video_file, manual=False)
print('frame count = ', frame_until)

bscan = np.zeros([length, frame_until - frame_start + 1, 3])
cap = cv2.VideoCapture(video_file)
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")
 
# play until video is completed
i = frame_start
while(cap.isOpened() and i < frame_until+1): 
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
    # Display the resulting frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bscan[:, i-frame_start] = extract_the_line(frame, p_1, p_2)
        if i==frame_start or i%100==0 or i==frame_until:
            print(i)
        i += 1
         
    # Press lower case "q" on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
 
    # Break the loop
    else:
        print(f'End of frames, i = {i}.')
        break
 

# when everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()


A = bscan + 0.000 # store old 2D image, just in case I might change its value in future accidently
if np.max(A, axis=None) > 1: # normalise the image
    bscan = (A - np.min(A, axis=None))/(np.max(A, axis=None) - np.min(A, axis=None))
    bscan[bscan > 1] = 1.0
    print('Frame values normalised to PNG.')
    
# line_rgb = bscan
fig = plt.figure(figsize=(16, 50), dpi=50)
if if_vertical == True: # vertical cutting
    plt.imshow(bscan)
else: # horizontal cutting
    # cv2.flip(line_rgb, -1) # rotate 180 degree
    # cv2.rotate(line_rgb, cv2.ROTATE_90_CLOCKWISE) # rotate 90 degree clockwise
    # cv2.rotate(line_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate 90 degree counter-clockwise
    plt.imshow(cv2.rotate(bscan, cv2.ROTATE_90_COUNTERCLOCKWISE))
    

plt.title("lines from all frames", fontsize=20)
plt.xlabel("X pixel scvaling", fontsize=20)
plt.ylabel("frame number", fontsize=20)
plt.show()

# later I can just load this data for post-processing from here on, instead of scanning through all frames again;    
mpimg.imsave(video_path+'2D_output_directly_from_video_'+video_name+'.png', bscan)   
    
###############################################################################

