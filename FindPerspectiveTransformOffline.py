import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


	#Fixme - compare this to what I ended up with
	#Pick Source Points on the image that we want to transform
	#X,Y
	#(598,447)..........(680,447)
	#   |					|
	#   |					|
	#(275,670)..........(1030,670)

	#Pick Destination Points - keep scaling correct
	# We could keep Y values the same for starters
	# and make upper X values match lower X values (Note: "lower" in image is higher since 0,0 top left of image)
	# This would look like this:
	#(275,447)..........(1030,447)
	#   |					|
	#   |					|
	#(275,670)..........(1030,670)

	#But then if I wanted to stretch it all the way to the top of the image I guess I could try making top Ys close to 0
	# This would look like this:
	#(275,100)..........(1030,100)
	#   |					|
	#   |					|
	#(275,670)..........(1030,670)	


os.chdir('C:/Users/bynum/documents/udacity/term1/dwb-t1-p2')
#img = plt.imread('test_images/trapazoid.jpg')
#img = plt.imread('test_images/roadliketrap.jpg')
#img = plt.imread('test_images/isolate_lane_pixels7.jpg')
img = plt.imread('test_images/test7.jpg')
#img = plt.imread('test_images/straight_lines1.jpg')


def transform_me(img):
	
	img_size = (img.shape[1], img.shape[0])
	print("image size: ", img_size)

	#Make a blank image to draw on off-line
	#blank = np.zeros_like(img)
	#cv2.imwrite('blank.jpg', blank)

	#define 4 source points src = np.float32([[,],[,],[,],[,]])
	# going in this order:  upper-right, lower-right, lower-left, upper-left
	case = '4'

	if(case == '1'):
		#case1 is simple/basic trapazoid in center of image
		src = np.float32([[1220,310],[1357,738],[646,738],[773,310]])
		dst = np.float32([[1357,310],[1357,738],[646,738],[646,310]])
	elif(case =='2'):
		#case2 is trapazoid that more closely resembles road image
		src = np.float32([[1056,670],[1600,1100],[400,1100],[936,670]])
		dst = np.float32([[1600,670],[1600,1100],[400,1100],[400,670]])
	elif(case =='3'):
		#case3 is straight_lines1
		src = np.float32([[734,482],[1025,665],[280,665],[548,482]])
		#src = np.float32([[671,440],[1025,665],[280,665],[606,440]])  #This was working
		#src = np.float32([[648,425],[1020,665],[277,665],[630,425]])	#This was working but not most recent
		
		#dst = np.float32([[675+300,440],[1020,665+54],[277,665+54],[606-300,440]]) #pushing it out slowly
		#dst = np.float32([[675+300,440-100],[1020,665+54],[277,665+54],[606-300,440-100]]) #pushing it out slowly 
		#dst = np.float32([[675+300,0],[1020,719],[277,719],[606-300,0]])
		dst = np.float32([[1025,0],[1025,719],[280,719],[280,0]]) #This was working
	elif(case =='4'):
		#case4 is for straight_lines1 as well but goes a little further up image to get larger portion of lane line
		#the further up you go the fuzzier it gets though because lower resolution or pixels/lane-line
		src = np.float32([[693,449],[1025,665],[280,665],[593,449]])
		dst = np.float32([[1025,0],[1025,719],[280,719],[280,0]])
	else:
		pass



	# print('shape of src is ', src.shape)
	# print('shape of dst is ', dst.shape)
	# print('src element', src[0,:])

	M = cv2.getPerspectiveTransform(src, dst)
	top_down = cv2.warpPerspective(img, M, img_size)


	#plt.imshow(img)

	fig, subplt = plt.subplots(1, 2, figsize=(18,7))
	fig.tight_layout()

	subplt[0].imshow(img)
	subplt[0].plot(src[0,0],src[0,1], '^')
	subplt[0].plot(src[1,0],src[1,1], '^')
	subplt[0].plot(src[2,0],src[2,1], '^')
	subplt[0].plot(src[3,0],src[3,1], '^')
	subplt[0].set_title('Original Image', fontsize=10)

	subplt[1].imshow(top_down)
	subplt[1].plot(dst[0,0],dst[0,1], '+')
	subplt[1].plot(dst[1,0],dst[1,1], '+')
	subplt[1].plot(dst[2,0],dst[2,1], '+')
	subplt[1].plot(dst[3,0],dst[3,1], '+')
	subplt[1].set_title('Top Down View', fontsize=10)
	plt.waitforbuttonpress(timeout=-1)
	plt.close('all')

	return()

transform_me(img)





	# original_image_points =np.zeros((4,1,2),np.float32)
	# desired_new_points =np.zeros((4,1,2),np.float32)


	# original_image_points[0,0,:] = 680,447
	# original_image_points[1,0,:] = 1030,670
	# original_image_points[2,0,:] = 275,670
	# original_image_points[3,0,:] = 598,447


	# desired_new_points[0,0,:] = img_size[0]*.75, img_size[1]*.25
	# desired_new_points[1,0,:] = img_size[0]*.75, img_size[1]*.75
	# desired_new_points[2,0,:] = img_size[0]*.25, img_size[1]*.75
	# desired_new_points[3,0,:] = img_size[0]*.25, img_size[1]*.25


	#original_image_points=np.float32([[680,447], [1030,670],[275,670],[598,447]])
	#desired_new_points=np.float32([[1030,447], [1030,670], [275,670],[275,447]])
    


	# original_image_points=np.float32(	[[680,447], 
	# 									 [1030,670],
	# 									 [275,670],
	# 									 [598,447]]    )
	# #desired_new_points=np.float32([[1030,447], [1030,670], [275,670],[275,447]])

	# desired_new_points=np.float32(		[[1120, 410], 
	# 									 [1120, 700], 
	# 									 [175, 700],
	# 									 [175, 410]]    )