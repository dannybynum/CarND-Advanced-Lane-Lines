[//]: # (Image References)
[image_BT_1]: ./output_images/roadliketrap.jpg  "Grayscale"
[image_cal_set]: ./camera_cal/Set_of_Checkerboard_Images.PNG  "Grayscale"

[image2]: ./camera_cal/undistort_output.png "Grayscale"
[image3]: ./output_images/isolate_lane_pixels4.jpg "Grayscale"
[image4]: ./output_images/top_down4.jpg "Grayscale"
[image5]: ./output_images/polyfitoutput4.jpg
[image6]: ./output_images/polyfitoutputwithtext4.jpg
[image7-8]: ./output_images/original_image_with_annotation4.jpg




<!-- This is the syntax for commenting/hiding text for readme/markdown -->
<!--[imageB]: ./images_in_writeup/Canny-fig.jpg "Grayscale" -->


## Danny's Term1 Project - Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Attempt to resize an image of full pipeline result:
<img src="https://github.com/dannybynum/DWB-T1-P2/blob/master/output_images/original_image_with_annotation4.jpg" width="48">

## Background / Scope of project
---

This was a very cool project that built directly on top of what was completed for the basic "lane finding" program created for the first project.  In this project more "real world" concerns had to be taken into account (e.g. camera lens distortion) as well as attempting to determine the curvature of the lane in addition to visually identifying the lane.

Interesting comment in instructions for project - shows how much emphasis is placed on ability to document your code:  
"In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project. "

---

### Structure of the code
The main pipeline is captured in the file [*Dannys_Lane_Line_Finder_For_Videos.py*](https://github.com/dannybynum/DWB-T1-P2/blob/master/Dannys_Lane_Line_Finder_For_Videos.py)

There is a file called [*Dannys_Lane_Line_Finder.py*](https://github.com/dannybynum/DWB-T1-P2/blob/master/Dannys_Lane_Line_Finder_For_Images.py) which I spent most of my time in to work out the pipeline on a set of 6 single frame test images that were provided.  It was useful to work on a set of possibly-problematic images in order to tune things so that they were at least at a good starting point for the video.  Since they were stills it was easy to switch between them and plot things side by side and stare at them or make lots of iterative tweaks.  I will keep this type of approach in mind for future efforts.

Within *Dannys_Lane_Line_Finder_For_Videos.py* there is a function called *proj2_process_image* which is written to recieve one image and output one image and uses similar video processing libraries as was used in the first project - here is a code snipet...
```python
def proj2_process_image (Current_Image):
	#Main pipeline steps all go here
	return Output_Image

output_video_filename = 'output_images/DannyProj2Video.mp4'
Input_Clip = VideoFileClip("project_video.mp4")
Processed_Clip = Input_Clip.fl_image(proj2_process_image) 
Processed_Clip.write_videofile(output_video_filename, audio=False, progress_bar = True)
```

The Functions are located in various files - generally grouped according to the different steps involved in the project.  The names are generally self explanatory.  Here is the order of function calls within the overall function: *proj2_process_image*:

* ImageEdgeTransforms.hls_select
* ImageEdgeTransforms.abs_sobel_thresh
* ImageEdgeTransforms.dir_threshold
* CurveFitFinder.slide_me
* CurveFitFinder.poly_fit_me
* CurveFitFinder.measure_curvature_pixels
* CurveFitFinder.measure_curvature_real

Several constants are listed in [*Config.py*](https://github.com/dannybynum/DWB-T1-P2/blob/master/Config.py)

Two steps occur prior to getting started with the main pipeline function.
* The Camera Calibration is all contained within [*CameraCalibration.py*](https://github.com/dannybynum/DWB-T1-P2/blob/master/Camera_Calibration.py) and the output of this is then manually copied (from terminal print out) into the *Config.py* file.
* The source and destination points for the perspective transform (from dash-cam to top-down/birds-eye view) are determined using the code and process in the file:  [*FindPerspectiveTransformOffline*](https://github.com/dannybynum/DWB-T1-P2/blob/master/FindPerspectiveTransformOffline.py) -- this is a manual guess and check process at the moment, and this file can be used with new images to manually come up with the right points.
* There is an optional file [*PlotLaneLineImageTransformsOffline*](https://github.com/dannybynum/DWB-T1-P2/blob/master/PlotLaneLineImageTransformsOffline.py) which can be used if needed to plot and tweak the parameters associated with the image transforms aimed at helping isolate the lane lines in the image.  This file provides a convenient way to tweak and plot -- but if values are changed here they would have to be manually copied back into the main pipeline in the *proj2_process_image*


### Building and Troubleshooting steps that helped me complete project
I built the entire project with processing the 6 sample images in several successive for loops -- in each case I was actually saving the files from the previous step and then reading them back in for the next step.  The code for this looks something like this:

```python
import glob
import os

os.chdir('C:/Users/bynum/documents/udacity/term1/dwb-t1-p2')

paths_of_test_imgs = glob.glob('test_images/test*.jpg')

image_count = 0

for sing_test_img_path in paths_of_test_imgs:
	image_count += 1 # Increment counter in for loop and use for save name
	fname = sing_test_img_path   
	img = cv2.imread(fname)

	undistorted_image = cv2.undistort(img, Config.cam_mtx, Config.dist_coeffs, None, Config.cam_mtx)
	save_name="test_images/"+"undistorted"+str(image_count)+".jpg"
	cv2.imwrite(save_name, undistorted_image)

print("Number of raw images that have been undistorted", image_count)
```

I struggled some with the image transformation step because no matter what I did the output (after transforming to "top down view/perspective") did not look like what I would have expected.  I noticed a BIG change (which now makes sense) depending on how far out (in real world range) I tried to go to grab the lane lines.  More could be said about this, but for the sake of brevity lets just say it was a huge help to just use powerpoint to create an image with a trapazoid that looked somewhat like the road and then playing with that to see what kind of output I got.  I also created and used the *FindPerspectiveTransformOffline.py*  script as a tool to help me plot/show the results of the transform as well as plotting the points that I was using as *source* and *destination* points.


![Image I created to help easily see top-down transform results][image_BT_1]
<!-- This =100x is setting WIDTH of image....can also set WIDTHxHEIGHT -->

### 1. The project pipeline involves the following steps

_Step 1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images_

Before correcting the distortion it has to be measured/determined.  In this case the method of measuring the distortion is to take images of an actual large checkerboard where the squares are precicely the same size.  Below is a screenshot of the thumbnails for the images used for this project:

![Camera Calibration / Measuring Distortion][image_cal_set]

Code snipets from my *CameraCalibration.py* file:
```python
# Read in each cal image in succession and if the inside corners arae found
# then append the list of image points to the overall set of image points
paths_of_cal_imgs = glob.glob('camera_cal/calibration*.jpg')
for sing_cal_img_path in paths_of_cal_imgs:
	#......

	# Convert to grayscale -- when using cv2.imread it comes in as BGR
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find the chessboard corners of each image
	corner_found_flag, corners = cv2.findChessboardCorners(img_gray, (nx, ny), None)
	#......
	#Append this list of image points for a single image to the overall set for this camer
  
    set_of_imgpts.append(corners)

    # Even though objpts is the same for each image we create a set here to match the size of the
    # set of detected image points to make the next step of creating the transformation  easier
    set_of_objpts.append(objpts)

#Then compute the distortion coefficients and camera matrix needed by using the set of points (across all checkerboard images)
cal_found_flag, cam_mtx, dist_coeffs, cam_rvecs, cam_tvecs = cv2.calibrateCamera(set_of_objpts, set_of_imgpts, img_shape, None, None)  
```


_Step 2: Apply a distortion correction to raw images_

There is a handy built in opencv function to remove the distortion from an image (in this case the distortion is where it "curves" some at the four corners) once you've "measured/determined" the distortion of the images (step1 above).  
```python
undistorted_image = cv2.undistort(test_image, cam_mtx, dist_coeffs, None, cam_mtx)
```

![Distortion Correction][image2]

_Step 3: Use color transforms, gradients, etc., to create a thresholded binary image_


```python
INSERT SOME CODE HERE
```


![Transforms to isolate lane pixels even with shadows on road][image3]


_Step 4: Apply a perspective transform to rectify binary image ("birds-eye view")_


```python
INSERT SOME CODE HERE
```

![Perspective transform to show top-down view of image][image4]


_Steps 5: Detect lane pixels and fit to find the lane boundary_

```python
INSERT SOME CODE HERE
```

![line fit using sliding windows][image5]

_Step 6: Determine the curvature of the lane and vehicle position with respect to center_


```python
INSERT SOME CODE HERE
```

![Overlay with lane curvature in pixels and meters][image6]


_Step 7: Warp the detected lane boundaries back onto the original image_
_Step 8: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position_


```python
INSERT SOME CODE HERE
```



![Warping the lane line fits back to original image space][image7-8]



### 2. Identify potential shortcomings with your current pipeline

The filters (Sobel,hsl-color) that are applied to extract the lane lines are fairly effective but looks like lots of additional work could be done here to get the optimal combination.

Tracking not implemented which causes the lane-overlay to "bounce" around some.  This could be implmemented and would be a big improvement.


### 3. Suggest possible improvements to your pipeline

Currently using the sliding window approach for _each_ image so the processing time is pretty long, go-take-a-walk long.  The course points out a cool way to use convolution to check for pixels directly to right and left of the fitted line and this would likely improve processing time and also make it more accurate.


__Future improvements could be made as follows__
1. See any "FIXME" present in the code - particularly in the main pipeline file "Dannys_Lane_Line_Finder_For_Videos" related to "tracking" and also related to the lane pixel extraction (see 'combo_4').
2. I did not test my pipeline on the 'challenge_video.mp4' or the 'harder_challenge.mp4' -- would be fun to do this -- use the "clip" feature that is currently commented out in the code.  It would also be cool to go take my own video and try it out on that.

