# -*- coding: utf-8 -*-
# @Author: Jiarui Wang(Jill)
# @Date:   2019-07-24 14:23:43
# @E-mail: e0386397@u.nus.edu
# @Last Modified time: 2019-07-26 15:54:55

import cv2 as cv
import cv2.structured_light as sl 
import numpy as np
import os

height, width = 64, 64
graycode = sl.GrayCodePattern_create(height, width)

pattern = graycode.generate()
pattern = np.array(pattern)
pattern = pattern[1]

blank_image = np.zeros((height,width,1), np.uint8)
white_image = np.ones((height,width,1), np.uint8)
white_image[:][:] = 255
# for i in range(height):
# 	for j in range(width):
# 		white_image[i][j] = 255

cv.imwrite("./pattern/pattern"+"black"+".jpg", blank_image)
cv.imwrite("./pattern/pattern"+"white"+".jpg", white_image)

print(len(pattern))
for i in range(len(pattern)):
	img1 = pattern[i]
	color_img = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
	h,w =img1.shape
	# chang the white into blue, and black into green
	# for m in range(h):
	# 	for n in range(w):
	# 		if color_img[m][n][0] == 0:
	# 			color_img[m][n] = [0, 0, 0]
	# 		else:
	# 			color_img[m][n] = [255, 255, 255]
	cv.imshow("img1", color_img)
	cv.imwrite("./pattern/pattern"+str(i)+".jpg", color_img)
	cv.waitKey()
	cv.destroyAllWindows()





# white = np.zeros([100,100,3],dtype=np.uint8)
# white.fill(255)
# cv.imshow("white", white)
# cv.waitKey()
# cv.destroyAllWindows()
# black = np.zeros([100,100,3],dtype=np.uint8)
# black.fill(0)
# cv.imshow("black", black)
# cv.waitKey()
# cv.destroyAllWindows()
# black, white = graycode.getImagesForShadowMasks(black, white)

# cv.imshow("black", black)
# cv.waitKey()
# cv.destroyAllWindows()