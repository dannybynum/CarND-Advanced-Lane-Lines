## Danny's Term1 Project - Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## Background / Scope of project
The course has videos and exercises that walk through examples of various computer vision functions that are needed to draw lines that represent the left and right lane markings.  Most of the code (but not all) for the project is already provided but you have to "tune" some of the parameters to make it produce the right results.



[//]: # (Image References)

[imageA]: ./images_in_writeup/DWB-writeup1.jpg "Grayscale"
[imageB]: ./images_in_writeup/Canny-fig.jpg "Grayscale"
[imageC]: ./images_in_writeup/ROIandRawLines.jpg
[imageD]: ./images_in_writeup/ImproveDrawLines.jpg


---

### Reflection

### 1. The project pipeline involves the following steps

_Step 0: Activate Environment and Notebook via Anaconda prompt_
Basic commands entered are as follows:
```
cd ~\Documents\Udacity\Term1>cd DWB-T1-P1
conda env list
activate carnd-term1
Jupyter Notebook
```

_Step 1: a step_


```
CODE
```

_Step 2: Image Operations to hide everything except edges_


![Image of Gaussian Blur][imageA]

```
Code
```

![Image of Canny Function Ouput][imageB]

_Step 3: Another step TODO_

```
TODO - insert some code here if desired.
```

The image below shows TODO:

![Image ROI and Raw Lines][imageC]

_Step 4: XYZ_

```
TODO - insert some code here if desired.
```

### 2. Identify potential shortcomings with your current pipeline

The filters (Sobel,hsl-color) that are applied to extract the lane lines are fairly effective but looks like lots of additional work could be done here to get the optimal combination.

Tracking not implemented which causes the lane-overlay to "bounce" around some.  This could be implmemented and would be a big improvement.


### 3. Suggest possible improvements to your pipeline

Currently using the sliding window approach for _each_ image so the processing time is pretty long, go-take-a-walk long.  The course points out a cool way to use convolution to check for pixels directly to right and left of the fitted line and this would likely improve processing time and also make it more accurate.


__Future improvements could be made as follows__
1. See any "FIXME" present in the code - particularly in the main pipeline file "Dannys_Lane_Line_Finder_For_Videos" related to "tracking" and also related to the lane pixel extraction (see 'combo_4').
2. I did not test my pipeline on the 'challenge_video.mp4' or the 'harder_challenge.mp4' -- would be fun to do this -- use the "clip" feature that is currently commented out in the code.  It would also be cool to go take my own video and try it out on that.

