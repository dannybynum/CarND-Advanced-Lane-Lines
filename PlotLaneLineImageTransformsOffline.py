import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

import os
import sys

import ImageEdgeTransforms
import ShowSideBySide




os.chdir('C:/Users/bynum/documents/udacity/term1/dwb-t1-p2/test_images')

#fname = 'test1.jpg'
fname = 'test2.jpg'  #s_channel not enough
#fname = 'test3.jpg'  #s_channel not enough
#fname = 'test4.jpg'
#fname = 'test5.jpg'  #Shadow Across Lane Line
#fname = 'test6.jpg'  #Shadow Across Lane Line
#fname = 'test7.jpg'   #s_channel not enough (this is actual a straight lines example I changed name to test7)




img_in = cv2.imread(fname)

img_in_RGB = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)


# Use hls_select to generate a binary image (1s,0s) corresponding to only the S-channel
# of the original image and then on top of that only S-channel values that fall within the threshold set 
#(i.e. pixels with certain intensity/brighntess) -- in this case anything above 100, or approx mid-range
s_binary_thresh, simg = ImageEdgeTransforms.hls_select(img_in,ch_num=2 ,thresh=(90, 255))

l_binary_thresh, l_img = ImageEdgeTransforms.hls_select(img_in,ch_num=1 ,thresh=(90, 255))

h_binary_thresh, h_img = ImageEdgeTransforms.hls_select(img_in,ch_num=0 ,thresh=(90, 255))

gray_binary_thresh, gray_img = ImageEdgeTransforms.hls_select(img_in,ch_num=99 ,thresh=(200, 255)) 

#ShowSideBySide.plot_me_gray(img_in, gray_img)

#ShowSideBySide.plot_me_gray(simg, s_binary_thresh)

# Take gradient with respect to x (vertical lines) and y (horizontal lines) separately which allows you
# to set different thresholds for each direction and then combine the results
# Note that this is similar to using the built in canny function in openCV but this gives you more control
gradx_binary = ImageEdgeTransforms.abs_sobel_thresh(simg, orient='x', thresh_min=20, thresh_max=150, sobel_kernel=9)
grady_binary = ImageEdgeTransforms.abs_sobel_thresh(simg, orient='y', thresh_min=60, thresh_max=150, sobel_kernel=9)

#The dir_threshold function is taking its own edge/sobel transform and then looking at what "direction" the lane lines are going in
#fixme - I'm not exactly sure what 0.8 to 1.2 translates to - I need to look back at this more - but seems to work
direction_binary = ImageEdgeTransforms.dir_threshold(simg, sobel_kernel=3, thresh=(0.8, 1.2))

#ShowSideBySide.plot_me_gray(gradx_binary, grady_binary)

# setup/prepare some blank images that you can use to step through the individual binary images and combine results
blank = np.zeros_like(simg)
combo_1 = np.zeros_like(simg)
combo_2 = np.zeros_like(simg)
combo_3 = np.zeros_like(simg)
combo_4 = np.zeros_like(simg)

# Combo1 is a bitwise AND of the thresholded gradient/sobel applied to x direction (vertical lines) and the overall direction threshold
combo_1[((gradx_binary == 1) & (direction_binary == 1))]= 1
# Combo1 is the same thing for the y direction (horizontal lines)
combo_2[((grady_binary==1) & (direction_binary ==1))] = 1
# Then using a bitwise OR to add x and y directions back together
combo_3[((gradx_binary == 1) | (grady_binary ==1))] = 255
# Then also add back in the color thresholding - do a bitwise and of combo3 and the color threshold binary
#combo_4[((combo_3==1) & (s_binary_thresh==1))] = 255  #make last combo the output combo equal to 255 so can see it on image easily

#Fixme - I'm overriding all of that with simple combination of X and Y gradients, I could/should use more tools here if possible
combo_4[((gradx_binary == 1) | (grady_binary ==1) | gray_binary_thresh ==1)] = 255






f, axs = plt.subplots(3, 3, figsize=(18,7))
f.tight_layout()

axs[0,0].imshow(img_in_RGB)  #, cmap = 'gray')
axs[0,0].set_title(str(fname), fontsize=10)

axs[0,1].imshow(simg, cmap='gray')
axs[0,1].set_title('simg', fontsize=10)

axs[0,2].imshow(s_binary_thresh, cmap='gray')
axs[0,2].set_title('s_binary_thresh', fontsize=10)

axs[1,0].imshow(gray_img, cmap='gray')
axs[1,0].set_title('gray_img', fontsize=10)


axs[1,1].imshow(gray_binary_thresh, cmap='gray')
axs[1,1].set_title('gray_binary_thresh', fontsize=10)

axs[1,2].imshow(combo_1, cmap='gray')
axs[1,2].set_title('combo_1', fontsize=10)

axs[2,0].imshow(combo_2, cmap='gray')
axs[2,0].set_title('combo_2')

axs[2,1].imshow(combo_3, cmap='gray')
axs[2,1].set_title('combo_3')

axs[2,2].imshow(combo_4, cmap='gray')
axs[2,2].set_title('combo_4')

plt.show()