# -*- coding: utf-8 -*-
# @Author: Jiarui Wang(Jill)
# @Date:   2019-06-13 11:21:39
# @E-mail: e0386397@u.nus.edu
# @Last Modified time: 2019-06-13 11:24:24
import cv2 as cv 
import numpy as np 
import sys


def on_mouse(event, x, y, flags, param):

	if event == cv.EVENT_LBUTTONDOWN:
		print("point", x, y)


img = cv.imread(sys.argv[1])
resizeAspect = 2
cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.resizeWindow('img', int(img.shape[1]/resizeAspect), int(img.shape[0]/resizeAspect))
cv.setMouseCallback('img', on_mouse)
cv.imshow('img', img)
cv.waitKey(0)