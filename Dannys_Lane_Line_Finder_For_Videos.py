# DWB,Udacity Self Driving Car Engineer Term1, Project 2 - Advanced Lane Line Finder
# Spent most of my time on the file "Dannys_Lane_Line_Finder" - working on a set of individual files
# Then that was able to be adapted pretty quickly to create a single function pipeline for application
# to the video - similar to how it was done for project 1.

#FIXME - note that the "Tracking" was suggested in tips and tricks has not been implemented, this would be
# a great addition but I'm not convinced it is required for project submission because my video looks pretty good
# even without it - so I will leave this out for now and plan to revisit sometime in the future


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

#%matplotlib inline

import os
import sys

import math
import pylab

import glob

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


#import CameraCalibration           # 
import ShowSideBySide              #Plot two Images side-by-side -- troubleshooting only
import ImageEdgeTransforms         #Apply sobel (gradient) transforms and also color transforms
import CurveFitFinder              #Fit lines to points and also generate polygon representing lane
import Config                      #Variables Used Across Modules



def proj2_process_image (Current_Image):
	###################################################################################################################
	# Step1:  Camera Calibration performed offline using file CameraCalibration.py, 
	# once performed these values are copied in Config.py file, which is called in this main pipeline

	###################################################################################################################
	# Step2:  Apply a distortion correction to raw images.

	#This uses the built in cv2 function and the previously calculated camera matrix and distortion coefficients
	undistorted_image = cv2.undistort(Current_Image, Config.cam_mtx, Config.dist_coeffs, None, Config.cam_mtx)

	###################################################################################################################
	# Step3:  Use color transforms, gradients, etc., to create a thresholded binary image.
	# Now it is time to perform some image transforms in order to make the pixels of the lane lines really stand out.
	# To do this we use some convenient color spaces (better than grayscale) and some gradient calculations.
	# I encapsulated these transforms in a module called ImageEdgeTransforms.py

	# Use hls_select to generate a binary image (1s,0s) corresponding to only the S-channel
	# of the original image and then on top of that only S-channel values that fall within the threshold set 
	#(i.e. pixels with certain intensity/brighntess) -- in this case anything above 100, or approx mid-range
	s_binary_thresh, hls_one_ch_img = ImageEdgeTransforms.hls_select(undistorted_image, ch_num=2 ,thresh=(90, 255))

	#ShowSideBySide.plot_me_gray(simg, s_binary_thresh)
    
	# Take gradient with respect to x (vertical lines) and y (horizontal lines) separately which allows you
	# to set different thresholds for each direction and then combine the results
	# Note that this is similar to using the built in canny function in openCV but this gives you more control
	gradx_binary = ImageEdgeTransforms.abs_sobel_thresh(hls_one_ch_img, orient='x', thresh_min=20, thresh_max=150, sobel_kernel=9)
	grady_binary = ImageEdgeTransforms.abs_sobel_thresh(hls_one_ch_img, orient='y', thresh_min=60, thresh_max=150, sobel_kernel=9)

	#The dir_threshold function is taking its own edge/sobel transform and then looking at what "direction" the lane lines are going in
	#fixme - I'm not exactly sure what 0.8 to 1.2 translates to - I need to look back at this more - but seems to work
	direction_binary = ImageEdgeTransforms.dir_threshold(hls_one_ch_img, sobel_kernel=3, thresh=(0.8, 1.2))

	gray_binary_thresh, gray_img = ImageEdgeTransforms.hls_select(undistorted_image,ch_num=99 ,thresh=(200, 255)) 

	#ShowSideBySide.plot_me_gray(gradx_binary, grady_binary)

	# setup/prepare some blank images that you can use to step through the individual binary images and combine results
	blank = np.zeros_like(hls_one_ch_img)
	combo_1 = np.zeros_like(hls_one_ch_img)
	combo_2 = np.zeros_like(hls_one_ch_img)
	combo_3 = np.zeros_like(hls_one_ch_img)
	combo_4 = np.zeros_like(hls_one_ch_img)

	# # Combo1 is a bitwise AND of the thresholded gradient/sobel applied to x direction (vertical lines) & overall dir thres
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


	#################################################################################################################
	#Step4: Create top-down view

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
	top_down = cv2.warpPerspective(combo_4, top_down_transform, Config.img_size)



	###############################################################################################################
	#Step5: Use Basic Sliding Window approach to detect the lane pixels in the image and fit it to an 
	
	#Fixme - The function calls below use "img_in" and "img_in_gray" -- no new information by copying top down 
	# to all three color channels so I'd like to come back and clean this up at some point
	img_in = np.dstack((top_down, top_down, top_down))
	#img_in_gray = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)
	img_in_gray = top_down
	
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

	img_with_fit, ploty, l_fit_pix, r_fit_pix, l_fit_real, r_fit_real, fill_pts = CurveFitFinder.poly_fit_me(leftx, 
																												lefty, 
																												rightx, 
																												righty, 
																												img_with_windows)

	################################################################################################################################
	#Step 6a (calculate curvature) happens within this loop - FIXME - correct labeling and/or flow to have it flow better with step5

	# Calculate the radius of curvature in pixels for both lane lines
	left_curverad_pix, right_curverad_pix = CurveFitFinder.measure_curvature_pixels(ploty, l_fit_pix, r_fit_pix)

	#Calculates the curvature of polynomial functions in meters for both lane lines
	left_curverad_real, right_curverad_real = CurveFitFinder.measure_curvature_real(ploty, l_fit_real, r_fit_real)

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

	################################################################################################################################
	# Step7: Warp the detected lane boundaries back onto the original image
	# from "tips and tricks for project"

	# Create an image to draw the lines on - from "tips and tricks for project"
	lane_visualize_canvas = np.zeros_like(img_with_fit_and_text2).astype(np.uint8)

	# Draw the lane onto the warped blank image, note fill_pts comes from fit_me function in ImageEdgeTransforms
	cv2.fillPoly(lane_visualize_canvas, np.int_([fill_pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	unwarp_lane_visualization = cv2.warpPerspective(lane_visualize_canvas, inverse_transform, Config.img_size) 

	###################################################################################################################
	# Step8: output a visual display of lanes and the curvature and centring text 
		
	orig_img_with_lane_fill = cv2.addWeighted(undistorted_image, 1, unwarp_lane_visualization, 0.3, 0)

	orig_with_fill_text1 = cv2.putText(orig_img_with_lane_fill, image_text_line1, (230, 50), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
	
	orig_with_fill_text2 = cv2.putText(orig_with_fill_text1, image_text_line2, (230, 75), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

	Output_Image = orig_with_fill_text2

	return Output_Image



###################################################################################################################
# Create Video, one image at a time by running pipeline on each image


# #Test Video Pipeline on single video 
# os.chdir('C:/Users/bynum/documents/udacity/term1/dwb-t1-p2/test_images')
# img = cv2.imread('test1.jpg')
# output = proj2_process_image(img)
# save_name = 'Test_video_Pipeline_Again.jpg'
# cv2.imwrite(save_name, output)


# Apply Video Pipeline to each image in video , and write new video

os.chdir('C:/Users/bynum/documents/udacity/term1/dwb-t1-p2')
output_video_filename = 'output_images/DannyProj2Video.mp4'
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
# ## Where start_second and end_second are integer values representing the start and end of the subclip
# ## You may also uncomment the following line for a subclip of the first 5 seconds
# ##Input_Clip = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
Input_Clip = VideoFileClip("project_video.mp4")
Processed_Clip = Input_Clip.fl_image(proj2_process_image) 
Processed_Clip.write_videofile(output_video_filename, audio=False, progress_bar = True)



####################################################################################################################
#Notes on future steps that could be applied to improve the pipeline:

# ~~~~~~~~~~~~~Tracking~~~~~~~~~~~~~~~~~
# After you've tuned your pipeline on test images, you'll run on a video stream, just like in the first project. In this case, however, you're going to keep track of things like where your last several detections of the lane lines were and what the curvature was, so you can properly treat new detections. To do this, it's useful to define a Line() class to keep track of all the interesting parameters you measure from frame to frame. Here's an example:

# # Define a class to receive the characteristics of each line detection
# class Line():
#     def __init__(self):
#         # was the line detected in the last iteration?
#         self.detected = False  
#         # x values of the last n fits of the line
#         self.recent_xfitted = [] 
#         #average x values of the fitted line over the last n iterations
#         self.bestx = None     
#         #polynomial coefficients averaged over the last n iterations
#         self.best_fit = None  
#         #polynomial coefficients for the most recent fit
#         self.current_fit = [np.array([False])]  
#         #radius of curvature of the line in some units
#         self.radius_of_curvature = None 
#         #distance in meters of vehicle center from the line
#         self.line_base_pos = None 
#         #difference in fit coefficients between last and new fits
#         self.diffs = np.array([0,0,0], dtype='float') 
#         #x values for detected line pixels
#         self.allx = None  
#         #y values for detected line pixels
#         self.ally = None  
# You can create an instance of the Line() class for the left and right lane lines to keep track of recent detections and to perform sanity checks.


# ~~~~~~~~~~~~~~~~~~~~~Sanity Check~~~~~~~~~~~~~~~~~~~~~~~~~~
# Ok, so your algorithm found some lines. Before moving on, you should check that the detection makes sense. To confirm that your detected lane lines are real, you might consider:

# Checking that they have similar curvature
# Checking that they are separated by approximately the right distance horizontally
# Checking that they are roughly parallel
# Look-Ahead Filter
# Once you've found the lane lines in one frame of video, and you are reasonably confident they are actually the lines you are looking for, you don't need to search blindly in the next frame. You can simply search within a window around the previous detection.

# For example, if you fit a polynomial, then for each y position, you have an x position that represents the lane center from the last frame. Search for the new line within +/- some margin around the old line center.

# If you need a reminder on how this works, make sure to go back and check the Finding the Lines: Search from Prior quiz from last lesson!

# Then check that your new line detections makes sense (i.e. expected curvature, separation, and slope).

# ~~~~~~~~~~~~~~~~~~~Reset~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If your sanity checks reveal that the lane lines you've detected are problematic for some reason, you can simply assume it was a bad or difficult frame of video, retain the previous positions from the frame prior and step to the next frame to search again. If you lose the lines for several frames in a row, you should probably start searching from scratch using a histogram and sliding window, or another method, to re-establish your measurement.

# ~~~~~~~~~~~~~~~~~~Smoothing~~~~~~~~~~~~~~~~~~~~~~~~~
# Even when everything is working, your line detections will jump around from frame to frame a bit and it can be preferable to smooth over the last n frames of video to obtain a cleaner result. Each time you get a new high-confidence measurement, you can append it to the list of recent measurements and then take an average over n past measurements to obtain the lane position you want to draw onto the image.

# ~~~~~~~~~~~~~~~~Drawing~~~~~~~~~~~~~~~~~~~~~~
# Once you have a good measurement of the line positions in warped space, it's time to project your measurement back down onto the road! Let's suppose, as in the previous example, you have a warped binary image called warped, and you have fit the lines with a polynomial and have arrays called ploty, left_fitx and right_fitx, which represent the x and y pixel values of the lines. You can then project those lines onto the original image as follows:

# # Create an image to draw the lines on
# warp_zero = np.zeros_like(warped).astype(np.uint8)
# color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# # Recast the x and y points into usable format for cv2.fillPoly()
# pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
# pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
# pts = np.hstack((pts_left, pts_right))

# # Draw the lane onto the warped blank image
# cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# # Warp the blank back to original image space using inverse perspective matrix (Minv)
# newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
# # Combine the result with the original image
# result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
# plt.imshow(result)
