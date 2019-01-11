import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

import os
import sys


#Color Transform Stuff
def hls_select(img, ch_num, thresh=(0, 255)):
    # use built in cv2 function to convert RGB or BGR to HLS color space
    # Could read in with either cv2.imread() or with mpimg.imread()
    # Use ~RGB2*NEWSPACE  with mpimg and ~BGR2*NEWSPACE with cv2
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Using this notation ([:,:,0]) to get only one color dimension - 0=Hue, 1=Level, 2=Saturation
    # The S channel seems to give good performance picking out lane lines regardless of line color and 
    # presence of shadows.  Other color spaces could also be explored and/or a combination.
    
    if ch_num==0:
        hls_one_ch_img = hls_img[:,:,0]
    elif ch_num==1:
        hls_one_ch_img = hls_img[:,:,1]
    elif ch_num==2:
        hls_one_ch_img = hls_img[:,:,2]
    else:
        hls_one_ch_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #print ("outputting grayscale image need to pass in a 'ch_num' to function 'hls_select'")
   
    #hls_thresh_bin = np.zeros_like(hls_img[:,:,0]) 
    hls_thresh_bin = np.zeros_like(hls_one_ch_img)

    #print(hls_thresh_bin.shape)
    
    #This looks like compact/clever way to set EACH element to 1 if it meets this comparison
    hls_thresh_bin[(hls_one_ch_img > thresh[0]) & (hls_one_ch_img <= thresh[1])] = 1
   
    # 3) Return a binary image of threshold result
    binary_output = hls_thresh_bin
    #binary_output = np.copy(img)
    return binary_output, hls_one_ch_img


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255, sobel_kernel=3):
    # Apply the following steps to img
    # 1) NOT converting to grayscale this time, using color spaces for better result
    #onech_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #passing in one channel of HLS space instead of grayscale
    onech_img = img

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    #note the cv2.Sobel(.....1,0) is x direction and ....0,1 is y direction
    if orient == 'x':  dir_sobel = cv2.Sobel(onech_img,cv2.CV_64F,1,0, ksize= sobel_kernel)

    elif orient == 'y':  dir_sobel = cv2.Sobel(onech_img,cv2.CV_64F,0,1, ksize= sobel_kernel)

    else:  sys.exit("Error - incorrect orient value passed into abs_sobel_thresh")

    # 3) Take the absolute value of the derivative or gradient
    abs_dir_sobel = np.absolute(dir_sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    #Note Dividing current value by max value gives number between 0 and 1
    scaled_sobel = np.uint8(255*abs_dir_sobel/np.max(abs_dir_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude 
    dir_sobl_binary = np.zeros_like(scaled_sobel)  # create blank image first
    
    #This looks like compact/clever way to set EACH element to 1 if it meets this comparison
    dir_sobl_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    
    #binary_output = np.copy(img) # lets function run/return without filling in pipeline
    binary_output = dir_sobl_binary

    return binary_output



# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
#def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    # Apply the following steps to img
        # 1) NOT  - Convert to grayscale - see belowS
    #onech_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #passing in one channel of HLS space instead of grayscale
    onech_img = img

    #2) Take the gradient in x and y separately
    #note the cv2.Sobel(.....1,0) is x direction and ....0,1 is y direction
    xsobel = cv2.Sobel(onech_img,cv2.CV_64F,1,0, ksize=sobel_kernel)

    ysobel = cv2.Sobel(onech_img,cv2.CV_64F,0,1, ksize=sobel_kernel)

    # 3) Calculate the magnitude (considering x and y)
    mag_sobel = np.sqrt(xsobel**2+ysobel**2)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    #Note Dividing current value by max value gives number between 0 and 1
    scaled_mag_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude 
    mag_sobl_binary = np.zeros_like(scaled_mag_sobel)  # create blank image first
    
    #This looks like compact/clever way to set EACH element to 1 if it meets this comparison
    mag_sobl_binary[(scaled_mag_sobel >= mag_thresh[0]) & (scaled_mag_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    
    #binary_output = np.copy(img) # lets function run/return without filling in pipeline
    xy_binary_output = mag_sobl_binary

    return xy_binary_output



def dir_threshold(img, sobel_kernel=3, thresh=(0, 1)):
    
    # Apply the following steps to img
        # 1) NOT  - Convert to grayscale - see belowS
    #onech_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #passing in one channel of HLS space instead of grayscale
    onech_img = img

    # 2) Take the gradient in x and y separately
    xsobel = cv2.Sobel(onech_img,cv2.CV_64F,1,0, ksize=sobel_kernel)

    ysobel = cv2.Sobel(onech_img,cv2.CV_64F,0,1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_xsobel = np.absolute(xsobel)

    abs_ysobel = np.absolute(ysobel)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    img_tangy = np.arctan2(abs_ysobel, abs_xsobel)
    # 5) Create a binary mask where direction thresholds are met
    img_tangy_binary = np.zeros_like(img_tangy)  # create blank image first
    
    #This looks like compact/clever way to set EACH element to 1 if it meets this comparison
    img_tangy_binary[(img_tangy >= thresh[0]) & (img_tangy <= thresh[1])] = 1
    

    # 6) Return this mask as your binary_output image
    binary_output = img_tangy_binary
    return binary_output


#Plot a histogram of the lower half of image (where lane lines are expected to be straight)
# In this case the "histogram" is really just summing the columns of pixels for each X position in the image
# and then plotting that sum across the image.
def hist(img):

    # Lane lines are likely to be mostly vertical nearest to the car
    # note could have left out the second part of first index ..."img.shape[0]" becuase just having ":"
    # after the first number would have the same result where it goes to the end of the index
    # bottom_half = img[img.shape[0]//2:img.shape[0],:]
    bottom_half = img[500:600,:]

    # The numpy "sum" function simplifies the process of summing across image pixels vertically
    # by typing the "axis=0" input to the funciton it knows to sum across y axis
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram