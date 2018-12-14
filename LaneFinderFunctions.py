import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os



os.chdir('C:/Users/bynum/documents/udacity/term1/DWB-T1-P2')

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
#The Distortion coefficients woulc be obtained by running something like this example:
#    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
#                                       imgpoints, gray.shape[::-1], None, None)

dist_pickle = pickle.load( open( "camera_cal/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

#print(mtx)
#print(dist)

# Read in an image
img_in = cv2.imread('camera_cal/calibration3.jpg')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y


print("This color_edges image is: ", type(img_in),
      "with dimensions: ", img_in.shape)



# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img_in, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist  - use cv2.undistort() function
                                       
    undist = cv2.undistort(img_in, mtx, dist, None, mtx) #img vs img_in

    # 2) Convert to grayscale
    undist_gray = cv2.cvtColor(undist,cv2.COLOR_BGR2GRAY)
    #undist_gray = cv2.cvtColor(undist,cv2.COLOR_RGB2GRAY)
    
    # 3) Find the chessboard corners

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(undist_gray, (nx, ny), None)

    #This code is a good reference to double check the dimensions of the
    #array you are working with
    ##print("x3 ndim: ", corners.ndim)
    ##print("x3 shape:", corners.shape)
    ##print("x3 size: ", corners.size)

    #x3 shape: (48, 1, 2)

    #print(corners[9,0,0])
    #print(corners[9,0,:])


    #ret=False
    # 4) If corners found: 
        #If found, draw corners
    if ret == True:
        # Draw the corners on the undistorted image, but still haven't performed perspective transform
        #fixme - should I be using img_in here??
        cv2.drawChessboardCorners(undist_gray, (nx, ny), corners, ret)
        #plt.imshow(img_out)  #can plot it but don't have to here

            # a) draw corners
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
        src=np.float32([corners[9,0,:],corners[14,0,:],corners[33,0,:],corners[38,0,:]])
        
        #Tip - this format also works for accessing the elements in the np-array
        #src=np.float32([corners[9],corners[14],corners[33],corners[38]])
        

        #print(src)

            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        img_size = (img_in.shape[1],img_in.shape[0])

        x_grid_sp = img_size[0]/nx
        y_grid_sp = img_size[1]/ny

        #improveMe - looks like you could use np.mgrid maybe

        #initialize an array of zeros of correct size
        dst_grid =np.zeros((4,2),np.float32)

        x_zm_val=200
        y_zm_val=(960/1280)*x_zm_val

        dst_grid[0:]=x_grid_sp*2, y_grid_sp*2
        dst_grid[1:]=x_grid_sp*7-x_zm_val, y_grid_sp*2

        dst_grid[2:]=x_grid_sp*2, y_grid_sp*5-y_zm_val
        dst_grid[3:]=x_grid_sp*7-x_zm_val, y_grid_sp*5-y_zm_val

        #######Note here is how "src" and "dst" were defined by instructor
        # offset = 100 # offset for dst points
        # img_size = (gray.shape[1], gray.shape[0])

        # # For source points I'm grabbing the outer four detected corners
        # src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # # For destination points, I'm arbitrarily choosing some points to be
        # # a nice fit for displaying our warped result 
        # # again, not exact, but close enough for our purposes
        # dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
        #                              [img_size[0]-offset, img_size[1]-offset], 
        #                              [offset, img_size[1]-offset]])
        ############################################################

        #print(dst_grid)

            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src,dst_grid)


            # e) use cv2.warpPerspective() to warp your image to a top-down view
            #Tip - very interesting - in quiz the image passed into the warpPerspective funciton
            #had to be the 960x1280x3 color (non-gray-scale image) or it would throw an error
        #warped = cv2.warpPerspective(undist_gray, M, img_size, flags=cv2.INTER_LINEAR)
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    else:
        print("Don't Trust This Output!!")
        dst_grid =np.zeros((4,2),np.float32)
        M = 52
        warped = undist_gray    

    return warped, M, dst_grid

#img_in = cv2.resize(img_in, (500, 400)


#cv2.imshow('display',img_in)
#cv2.waitKey(0)

top_down, perspective_M, dst_grid = corners_unwarp(img_in, nx, ny, mtx, dist)

img_in_gray= cv2.cvtColor(img_in,cv2.COLOR_RGB2GRAY)

#Note here is how you could plot the points of the dst_grid to see where they are being placed.
#Tip - its a 4x2 array so I can access it by [row][column] where column is x,y ->0,1
# plt.imshow(top_down)
# plt.plot(dst_grid[0][0],dst_grid[0][1],'.')
# plt.plot(dst_grid[1][0],dst_grid[1][1],'.')
# plt.plot(dst_grid[2][0],dst_grid[2][1],'.')
# plt.plot(dst_grid[3][0],dst_grid[3][1],'.')
# plt.show()





#output plot
plt.ioff() 
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
f.tight_layout()
ax1.imshow(img_in)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(top_down,cmap='gray')
ax2.set_title('Undistorted and Warped Image', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()