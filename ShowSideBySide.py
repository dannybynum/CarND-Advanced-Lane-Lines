import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def plot_me_gray(original_img, manipulated_img):
    fig, subplt = plt.subplots(1, 2, figsize=(18,7))
    fig.tight_layout()

    subplt[0].imshow(original_img, cmap = 'gray')
    subplt[0].set_title('Left Image', fontsize=10)

    subplt[1].imshow(manipulated_img , cmap = 'gray')
    subplt[1].set_title('Right Image', fontsize=10)
    plt.waitforbuttonpress(timeout=-1)
    plt.close('all')

    return()


def inspectimage():
	os.chdir('C:/Users/bynum/documents/udacity/term1/dwb-t1-p2')
	img = plt.imread('test_images/straight_lines1.jpg')
	plt.imshow(img)
	
	img_size = 1280,720

	plt.plot(680,447,'.')
	plt.plot(1030,670,'+')
	plt.plot(275,670,'<')
	plt.plot(598,447,'>')

	plt.plot(1120,410,'.')
	plt.plot(1120,700,'+')
	plt.plot(175,700,'<')
	plt.plot(175,410,'>')

	
	plt.waitforbuttonpress(timeout=-1)
	plt.close('all')

	return()

#inspectimage()



# src=np.float32([(598,447), (680,447), (275,670), (1030,670)])

# dst=np.float32([(275,100), (1030,100), (275,670), (1030,670)])

# print("src",src)
# print("dst",dst)