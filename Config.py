import cv2
import os
import numpy as np
import CameraCalibration

#Variables used throughout
Count_Frames_Processed = 0
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension


# NOTE - you only have to run the CameraCalibration.calibrate_me() function one time then you can copy the values in as I did below
#  this would be repeated if you change cameras and need to redo the calibration

# cam_mtx, dist_coeffs, notused1, undistorted_img = CameraCalibration.calibrate_me()
# print(cam_mtx)
# print(dist_coeffs)

# These values were copied in from running the CameraCalibration.calibrate_me() function on the set of test images
cam_mtx =  np.array([[  1.15396093e+03,   0.00000000e+00,   6.69705359e+02],
 [  0.00000000e+00,   1.14802495e+03,   3.85656232e+02],
 [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

dist_coeffs = np.array([ -2.41017968e-01,  -5.30720497e-02,  -1.15810318e-03,  -1.28318543e-04,  2.67124302e-02])


os.chdir('C:/Users/bynum/documents/udacity/term1/dwb-t1-p2/test_images')

#basing image sizes and blank images off of test1.jpg throughout project
img_in = cv2.imread('test1.jpg')

img_width = img_in.shape[1]
img_height = img_in.shape[0]

img_size = (img_width, img_height)


#Note with these source / original_image points, the transform looks good, but I may be giving up
#some information because it cuts off at a closer distance (lower in image) - see plot with "FindPerspectiveTransformOffline.py"
#original_image_points = np.float32([[734,482],[1025,665],[280,665],[548,482]])
original_image_points = np.float32([[693,449],[1025,665],[280,665],[593,449]])

desired_new_points = np.float32([[1025,0],[1025,719],[280,719],[280,0]])



#Create default "previously_used_fill_points" variable
#########################################
ploty = np.linspace(0, img_in.shape[0]-1, img_in.shape[0] )
left_fitx = 1*ploty**2 + 1*ploty
right_fitx = 1*ploty**2 + 1*ploty
fill_pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
fill_pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
previously_used_fill_pts1 = np.hstack((fill_pts_left, fill_pts_right))
previously_used_fill_pts2 = np.hstack((fill_pts_left, fill_pts_right))

lane_width_at_eval_pixels = 0