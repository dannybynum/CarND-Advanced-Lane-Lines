import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

import Config

# Load our image
#top_down = mpimg.imread('warped_example.jpg')

def slide_me(top_down, leftx_start, rightx_start):
    
    leftx_current = leftx_start
    rightx_current = rightx_start

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(top_down.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = top_down.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Not needed - see note below - originally used np.dstack to create a 3-channel/color 
    #  output image to draw on and visualize the result
    # Note - do not use this "np.dstack" fucntion because when saving and reading in with open CV you get a 3 channel image anyway
    # Even though with the last operation I did on this image (top down view) it is just a single channel 
    # - save and/or read-in converts back to 3-channel
    #out_img = np.dstack((img_in, img_in, img_in))

    out_img = top_down

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = top_down.shape[0] - (window+1)*window_height
        win_y_high = top_down.shape[0] - window*window_height
        ### DWB Done (was to-do): Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        #Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### DWB-DONE: Identify the nonzero pixels in x and y within the window ###
        # DWB_Note: already have non-zero so just have to confine it to the window
        TF_array_good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high))

        good_left_inds = TF_array_good_left_inds.nonzero()[0]
        
        
        TF_array_good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high))

        good_right_inds = TF_array_good_right_inds.nonzero()[0]

        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        # and make sure we don't recenter the window when there is a gap in lane line
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    # For example "left_lane_inds" is the overall element count and nonzerox has the
    # actual x coordinate
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def poly_fit_me(leftx, lefty, rightx, righty, img_in):
    
    # Find our lane pixels first
    #leftx, lefty, rightx, righty, out_img = find_lane_pixels(top_down)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    #This fits a polynomial in real units (meters) for calculating curvature -this on is not plotted visually
    left_fit_real = np.polyfit(lefty*Config.ym_per_pix, leftx*Config.xm_per_pix, 2)
    right_fit_real = np.polyfit(righty*Config.ym_per_pix, rightx*Config.xm_per_pix, 2)
    

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_in.shape[0]-1, img_in.shape[0] )


    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    img_in[lefty, leftx] = [255, 0, 0]
    img_in[righty, rightx] = [0, 0, 255]

    # Fixme - got this from web and need to spend more time understanding what it does
    # the note from online mentions that it - Recast the x and y points into usable format for cv2.fillPoly()
    left_pts = np.vstack((left_fitx,ploty)).astype(np.int32).T
    right_pts = np.vstack((right_fitx,ploty)).astype(np.int32).T

    img_in_with_line_left = cv2.polylines(img_in,[left_pts],False,(0,255,255))
    img_in_with_lines = cv2.polylines(img_in_with_line_left,[right_pts],False,(0,255,255))
    

    # Don't need this but cuold be useful for plotting/t-shooting
    # Using hte polylines function to draw the line on the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')


    #calculate the array needed for fillpoly using suggested transforms from project tips and tricks
    # Recast the x and y points into usable format for cv2.fillPoly()
    fill_pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    fill_pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    fill_pts = np.hstack((fill_pts_left, fill_pts_right))


    # blank = np.zeros_like(img_in_with_lines).astype(np.uint8)
    # lane_visualize_canvas = np.dstack((blank, blank, blank))

    # cv2.fillPoly(lane_visualize_canvas, np.int_([fill_pts]), (0,255, 0))

    return img_in_with_lines, ploty, left_fit, right_fit, left_fit_real, right_fit_real, fill_pts


# out_img = fit_polynomial(top_down)

# plt.imshow(out_img)


    
def measure_curvature_pixels(ploty, left_fit, right_fit):
    
    #Calculates the curvature of polynomial functions in pixels.
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    ## Implement the calculation of the left line here
    left_radi_pix = ((1+(2*left_fit[0]*y_eval+left_fit[1])**2)**(3/2)) /(2*np.absolute(left_fit[0])) 
    
    ## Implement the calculation of the right line here
    right_radai_pix = ((1+(2*right_fit[0]*y_eval+right_fit[1])**2)**(3/2)) /(2*np.absolute(right_fit[0]))  
    
    
    

    return left_radi_pix, right_radai_pix


def measure_curvature_real(ploty, left_fit, right_fit): 
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature) - using formula from Udacity lecture
    left_radi_rl = ((1 + (2*left_fit[0]*y_eval*Config.ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_radi_rl = ((1 + (2*right_fit[0]*y_eval*Config.ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_radi_rl, right_radi_rl


#Note - this is not used currently
def write_radi_me(img_in):
    
    img_out = cv2.putText(img, "This one!", (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return img_out