# DWB,Udacity Self Driving Car Engineer Term1, Project 2 - Advanced Lane Line Finder
# Started work Dec-16,2018 took some breaks, finished on Jan-11-2019
# Steps/Requirements for project are as follows:

# Step1:  Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# Step2:  Apply a distortion correction to raw images.
# Step3:  Use color transforms, gradients, etc., to create a thresholded binary image.
# Step4:  Apply a perspective transform to rectify binary image ("birds-eye view").
# Step5:  Detect lane pixels and fit to find the lane boundary.
# Step6:  Determine the curvature of the lane and vehicle position with respect to center.
# Step7:  Warp the detected lane boundaries back onto the original image.
# Step8:  Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

import os
import sys

import math
import pylab

import glob

#import LaneFinderFunctions 
import CameraCalibration           # 
import ShowSideBySide              #Plot two Images side-by-side -- troubleshooting only
import ImageEdgeTransforms         #Apply sobel (gradient) transforms and also color transforms
import CurveFitFinder              #Fit lines to points and also generate polygon representing lane
import Config                      #Variables Used Across Modules


###################################################################################################################
# Step1:  Compute camera calibration matrix and distortion coefficients given a set of chessboard images
# obtain camera matrix and distortion coefficients using a set of calibration images and several openCV built in functions
# these built in functions area all contained within a custom function/def I wrote called "CalibrateMe" in the CameraCalibration.py module.
# The primary openCV built in functions that are used are:  cv2.findChessboardCorners and cv2.calibrateCamera


###################################################################################################################
# Step1:  Camera Calibration performed offline, outside of this main pipeline and values copied into the Config.py file



###################################################################################################################
# Step2:  Apply a distortion correction to raw images.
os.chdir('C:/Users/bynum/documents/udacity/term1/dwb-t1-p2')

paths_of_test_imgs = glob.glob('test_images/test*.jpg')
image_count = 0


# Read in each cal image in succession and if the inside corners arae found
# then append the list of image points to the overall set of image points so 
# we can use them to calcualte a camera distortion matrix
for sing_test_img_path in paths_of_test_imgs:
	image_count += 1 # Increment counter in for loop and use for save name

	fname = sing_test_img_path
    
	img = cv2.imread(fname)

	#This uses the built in cv2 function and the previously calculated camera matrix and distortion coefficients
	#and since it's inside the for loop it performs it on each image in the test_images directory
	undistorted_image = cv2.undistort(img, Config.cam_mtx, Config.dist_coeffs, None, Config.cam_mtx)

	save_name="test_images/"+"undistorted"+str(image_count)+".jpg"

	cv2.imwrite(save_name, undistorted_image)

	# If you want to visulaize the undistort result then uncomment these lines
	# ShowSideBySide.plot_me(img, undistorted_image)  # Fixme, this is plotting BRG ans RGB so colors look 
													#    off but don't care right now

print("Number of raw images that have been undistorted", image_count)

###################################################################################################################
# Step3:  Use color transforms, gradients, etc., to create a thresholded binary image.
# Now it is time to perform some image transforms in order to make the pixels of the lane lines really stand out.
# To do this we use some convenient color spaces (better than grayscale) and some gradient calculations.
# I encapsulated these transforms in a module called ImageEdgeTransforms.py


paths_of_undist_imgs = glob.glob('test_images/undistorted*.jpg')
image_count = 0

for sing_undist_img_path in paths_of_undist_imgs:

	image_count += 1  # Increment counter in for loop and use for save name

	fname = sing_undist_img_path
	img_in = cv2.imread(fname)

	# Use hls_select to generate a binary image (1s,0s) corresponding to only the S-channel
	# of the original image and then on top of that only S-channel values that fall within the threshold set 
	#(i.e. pixels with certain intensity/brighntess) -- in this case anything above 100, or approx mid-range
	s_binary_thresh, hls_one_ch_img = ImageEdgeTransforms.hls_select(img_in, ch_num=2 ,thresh=(90, 255))

	#ShowSideBySide.plot_me_gray(simg, s_binary_thresh)
    
	# Take gradient with respect to x (vertical lines) and y (horizontal lines) separately which allows you
	# to set different thresholds for each direction and then combine the results
	# Note that this is similar to using the built in canny function in openCV but this gives you more control
	gradx_binary = ImageEdgeTransforms.abs_sobel_thresh(hls_one_ch_img, orient='x', thresh_min=20, thresh_max=150, sobel_kernel=9)
	grady_binary = ImageEdgeTransforms.abs_sobel_thresh(hls_one_ch_img, orient='y', thresh_min=60, thresh_max=150, sobel_kernel=9)

	#The dir_threshold function is taking its own edge/sobel transform and then looking at what "direction" the lane lines are going in
	#fixme - I'm not exactly sure what 0.8 to 1.2 translates to - I need to look back at this more - but seems to work
	direction_binary = ImageEdgeTransforms.dir_threshold(hls_one_ch_img, sobel_kernel=3, thresh=(0.8, 1.2))

	gray_binary_thresh, gray_img = ImageEdgeTransforms.hls_select(img_in,ch_num=99 ,thresh=(200, 255)) 

	#ShowSideBySide.plot_me_gray(gradx_binary, grady_binary)

	# setup/prepare some blank images that you can use to step through the individual binary images and combine results
	blank = np.zeros_like(hls_one_ch_img)
	combo_1 = np.zeros_like(hls_one_ch_img)
	combo_2 = np.zeros_like(hls_one_ch_img)
	combo_3 = np.zeros_like(hls_one_ch_img)
	combo_4 = np.zeros_like(hls_one_ch_img)

	# # Combo1 is a bitwise AND of the thresholded gradient/sobel applied to x direction (vertical lines) and the overall direction threshold
	# combo_1[((gradx_binary == 1) & (direction_binary == 1))]= 1
	# # Combo1 is the same thing for the y direction (horizontal lines)
	# combo_2[((grady_binary==1) & (direction_binary ==1))] = 1
	# # Then using a bitwise OR to add x and y directions back together
	# combo_3[((combo_1==1) | (combo_2==1))] = 1
	# # Then also add back in the color thresholding - do a bitwise and of combo3 and the color threshold binary
	# #combo_4[((combo_3==1) & (s_binary_thresh==1))] = 255  #make last combo the output combo equal to 255 so can see it on image easily

	#Fixme - I'm overriding all of that with simple combination of X and Y gradients, I could/should use more tools here if possible
	#Fixme - on Jan 9th added in the gray_binary_thresh which skips all the gradient transforms - may be dangerous
	combo_4[((gradx_binary == 1) | (grady_binary ==1) | gray_binary_thresh ==1)] = 255

    # This stacking of multiple binary images can help you visualize the results of combining with and or 'or'
    # This creates a red=off, green=sxbinary, Blue=s_binary, image so you can
    # see the contributions from the different elements on the same image if yellow then both there (red+green)
	#visualize_with_colors_same_img =  np.dstack((blank, combo_1, combo_3))* 255

	#ShowSideBySide.plot_me_gray(combine_xy, combinexy_and_color)
	#ShowSideBySide.plot_me_gray(s_binary_thresh, combo_4)

	#print("combo_4 shape", combo_4.shape)


	save_name="test_images/"+"isolate_lane_pixels"+str(image_count)+".jpg"

	cv2.imwrite(save_name, combo_4)


print("Number of images where lane pixels have been isolated", image_count)



###################################################################################################################
# Step4: Create top-down view

paths_of_lane_pixels_imgs = glob.glob('test_images/isolate_lane_pixels*.jpg')

#to create color images showing transform do this but be careful not to leave this way
#paths_of_lane_pixels_imgs = glob.glob('test_images/undistorted*.jpg')
image_count = 0

for sing_lane_pixels_img_path in paths_of_lane_pixels_imgs:


	image_count += 1  # Increment counter in for loop and use for save name

	fname = sing_lane_pixels_img_path
	img_in = cv2.imread(fname)
    

	#original_image_points = np.float32([[671,440],[1025,665],[280,665],[606,440]])

	#Note with these source / original_image points, the transform looks good, but I may be giving up
	#some information because it cuts off at a closer distance (lower in image) - see plot with "FindPerspectiveTransformOffline.py"
	#original_image_points = np.float32([[734,482],[1025,665],[280,665],[548,482]])
	original_image_points = np.float32([[693,449],[1025,665],[280,665],[593,449]])

	desired_new_points = np.float32([[1025,0],[1025,719],[280,719],[280,0]])


    #Using built in function to calculate the perspective transform given source and destination points
	top_down_transform = cv2.getPerspectiveTransform(original_image_points, desired_new_points)

	inverse_transform = cv2.getPerspectiveTransform(desired_new_points, original_image_points)


    #Using built in function to perform transform given transform matrix calculated above
	#top_down = cv2.warpPerspective(gray_in, M, img_size, flags=cv2.INTER_LINEAR)
	top_down = cv2.warpPerspective(img_in, top_down_transform, Config.img_size)

	save_name="test_images/"+"top_down"+str(image_count)+".jpg"
	#save_name="test_images/"+"color_top_down"+str(image_count)+".jpg"

	cv2.imwrite(save_name, top_down)

print("Number of images transformed to top-down view", image_count)



###################################################################################################################
# Steps5-7 are in this next for loop
# Step5: Use Basic Sliding Window approach to detect the lane pixels in the image and fit it to an 
#  equation description of the lane lines
#  Note I like the Basic Sliding Window because it is very apparent what algorithm/approach is being used
#  Fixme - I may implement the convolution method after successfully showing the sliding window approach

paths_of_top_down_imgs = glob.glob('test_images/top_down*.jpg')
image_count = 0

for sing_top_down_img_path in paths_of_top_down_imgs:

	image_count += 1  # Increment counter in for loop and use for save name

	fname = sing_top_down_img_path
	img_in = cv2.imread(fname)

	img_in_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
	
	# Take a histogram of the bottom half of the image
	histogram = np.sum(img_in_gray[img_in_gray.shape[0]//2:,:], axis=0)

	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]//2)  #result of all of this is just '640'

    # note that argmax does -- Returns the indices of the maximum values along an axis.
    # pretty cool - search for the x position (i.e "index" of the value) for value at max 
    # using the argmax function and searching from 0 to midpoint and from midpoint to end of image
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Current positions to be updated later for each window in nwindows
	leftx_start = leftx_base
	rightx_start = rightx_base



	#Function call to place windows on the image starting on the bottom (based on histogram) and going
	# all the way up the image - shifting left and right based on the median pixel position in that 
	# particular window.

	leftx, lefty, rightx, righty, img_with_windows = CurveFitFinder.slide_me(img_in, leftx_start, rightx_start)


	# Function call to fit a polynomial to the pixels found within the sliding search window
	# that results from the previous function call (slide_me)

	img_with_fit, ploty, left_fit_pix, right_fit_pix, left_fit_real, right_fit_real, fill_pts = CurveFitFinder.poly_fit_me(leftx, lefty, rightx, righty, img_with_windows)

	################################################################################################################################
	#Step 6a (calculate curvature) happens within this loop - FIXME - correct labeling and/or flow to have it flow better with step5

	# Calculate the radius of curvature in pixels for both lane lines
	left_curverad_pix, right_curverad_pix = CurveFitFinder.measure_curvature_pixels(ploty, left_fit_pix, right_fit_pix)

	#Calculates the curvature of polynomial functions in meters for both lane lines
	left_curverad_real, right_curverad_real = CurveFitFinder.measure_curvature_real(ploty, left_fit_real, right_fit_real)

	#print("Image Number ",image_count, "Pixel Curvature: ", left_curverad_pix, right_curverad_pix)
	#print("Image Number ",image_count, "Real Curvature (meters): ", left_curverad_real, right_curverad_real)
	# Should see values of 1625.06 and 1976.30 here, if using
	# the default `generate_data` function with given seed number

	
	#calculate the average curvature between left and right lines
	curve_print = (left_curverad_real + right_curverad_real)/2



	################################################################################################################################
	#Step 6b - calculate how centered the vehicle is in the lane.
	
	#Calculate centering based on how close the center of the lane lines is to the center of the image
	lane_center_pixels = (img_in_gray.shape[1]/2) -(rightx_start - leftx_start)

	lane_center_meters = lane_center_pixels*Config.xm_per_pix

	
	image_text_line1="Lane Curvature " + str(round(curve_print,1)) + " meters"
	image_text_line2="Distance From Lane Center " + str(round(lane_center_meters,1)) + " meters"

	#writing text on the images using the cv2 'putText' function
	font = cv2.FONT_HERSHEY_SIMPLEX
	img_with_fit_and_text = cv2.putText(img_with_fit, image_text_line1, (230, 50), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
	
	img_with_fit_and_text2 = cv2.putText(img_with_fit_and_text, image_text_line2, (230, 75), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

	save_name="test_images/"+"polyfitoutputwithtext"+str(image_count)+".jpg"

	cv2.imwrite(save_name, img_with_fit_and_text2)

	################################################################################################################################
	# Step7: Warp the detected lane boundaries back onto the original image
	# from "tips and tricks for project"

	# Create an image to draw the lines on - from "tips and tricks for project"
	# blank = np.zeros_like(img_with_fit_and_text2).astype(np.uint8)
	# lane_visualize_canvas = np.dstack((blank, blank, blank))
	lane_visualize_canvas = np.zeros_like(img_with_fit_and_text2).astype(np.uint8)

	#print("fill pts shape for image ",image_count," ", fill_pts.shape)

	# Draw the lane onto the warped blank image, note fill_pts comes from fit_me function in ImageEdgeTransforms
	cv2.fillPoly(lane_visualize_canvas, np.int_([fill_pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	# Fixme - Don't need to use shape.  I think image size is already defined, several examples for this type of thing that could be cleaned up
	unwarp_lane_visualization = cv2.warpPerspective(lane_visualize_canvas, inverse_transform, Config.img_size) 
	

	# Combine the result with the original image
	#get original image - not being read in for this loop but can be read in using paths already created for a different loop
	fname = paths_of_test_imgs[int(image_count-1)]
	orig_img_in = cv2.imread(fname)

	# print("unwarp_lane_visualization shape ",image_count," ", unwarp_lane_visualization.shape)
	# print("img_in ",image_count," ", img_in.shape)
	
	orig_img_with_lane_fill = cv2.addWeighted(orig_img_in, 1, unwarp_lane_visualization, 0.3, 0)

	orig_with_fill_text1 = cv2.putText(orig_img_with_lane_fill, image_text_line1, (230, 50), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
	
	orig_with_fill_text2 = cv2.putText(orig_with_fill_text1, image_text_line2, (230, 75), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

	save_name="test_images/"+"orignal_image_with_annotation"+str(image_count)+".jpg"

	cv2.imwrite(save_name, orig_with_fill_text2)




print("Number of images with polynomial fitted, Curvature/Center Calculated, and warped back to original", image_count)



###################################################################################################################
# Step8: output a visual display of lanes and the curvature and centring text --- this is already in place incrementally in other steps
#no addtional code needed for this step at this time