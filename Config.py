import cv2
import os
import numpy as np
import CameraCalibration

#Variables used throughout
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

img_size = (img_in.shape[1], img_in.shape[0])