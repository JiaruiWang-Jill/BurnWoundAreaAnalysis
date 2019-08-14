# -*- coding: utf-8 -*-
# @Author: Jiarui Wang(Jill)
# @Date:   2019-06-08 11:42:26
# @E-mail: e0386397@u.nus.edu
# @Last Modified time: 2019-07-08 11:31:01
import cv2 as cv 
import numpy as np 
import sys
import math
import operator
import scipy
import os
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from numpy.linalg import inv
from scipy.spatial import Delaunay
########################################################################
#################################################################
# This is for estimating area in real world according to image. 
# By building homography martrix
################################################################
########################################################################


resizeAspect = 2
manualCorners=[]
contourPoints=[]
global min_x, min_y, width, height, point1, point2, ix, iy
global ROI
global img 
global shapeIndex


########################################################################
# Sign real coordinates to corners
########################################################################
def getRealCoordinateList(times):
	patternPoints = []
	t = 1
	for i in range(times):
		for i in range(0,10):
			patternPoints.append([0, t*(25+10*i), 0])

		for i in range(0,10):
			patternPoints.append([t*10, t*(25+10*i), 0])

		for i in range(0,10):
			patternPoints.append([t*(15+10*i), t*10, 0])

		for i in range(0,10):
			patternPoints.append([t*(15+10*i), 0, 0])

	return patternPoints

########################################################################
# Use Ap=b to calculate homography matrix
########################################################################
def calculatehomographyMatrix(pixelList, realList):
	# Get matrix A from pixel(u,v) and real (x,y)
	A = []
	b = []
	length =  len(pixelList)
	for i in range(0, length):
		u = pixelList[i][0]
		v = pixelList[i][1]
		x = realList[i][0]
		y = realList[i][1]
		A.append([x, y, 1, 0, 0, 0, -1*u*x, -1*u*y])
		b.append([u])
	for i in range(0, length):
		u = pixelList[i][0]
		v = pixelList[i][1]
		x = realList[i][0]
		y = realList[i][1]
		A.append([0, 0, 0, x, y, 1, -1*v*x, -1*v*y])
		b.append([v])

	A = np.array(A)
	b = np.array(b)
	p, residuals, rank, sv = np.linalg.lstsq(A, b, -1)


	homography = [ [p[0][0], p[1][0], p[2][0]],
				   [p[3][0], p[4][0], p[5][0]],
				   [p[6][0], p[7][0], 1]]
	homography = np.array(homography)

	return homography


########################################################################
# Find countours of detected shape
########################################################################
def findContourPoints():
	contourPoints.clear()

	global min_x, min_y, width, height
	global ROI, manualCorners,img
	manualCorners.clear()
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	# select region of interest, so that corner detection can be more accurate
	cv.namedWindow('temp', cv.WINDOW_NORMAL)
	cv.resizeWindow('temp', int(img.shape[1]/resizeAspect), int(img.shape[0]/resizeAspect))
	cv.setMouseCallback('temp', on_mouse)
	cv.imshow('temp', img)
	cv.waitKey(0)
	print("get region of interest ...")

	img2 = img.copy()
	ROI = img2[min_y:min_y+height, min_x:min_x+width]

	cv.namedWindow('ROI', cv.WINDOW_NORMAL)
	cv.resizeWindow('ROI', int(ROI.shape[1]/resizeAspect), int(ROI.shape[0]/resizeAspect))
	cv.setMouseCallback('ROI', on_mouse4)
	while(1):
		cv.imshow('ROI', ROI)
		k = cv.waitKey(20)&0xFF
		if k==27:
			break
	cv.destroyAllWindows()
	print("----test----, in find contour points ", contourPoints)
	points = np.array(contourPoints)

	return points


########################################################################
# Calculate contour Area.
########################################################################
def calculateContourArea(pixelPoints, homography):
	global shapeIndex
	realPoints = []
	# Get triangles
	for point in pixelPoints:
		p = np.array([point[0], point[1], 1])
		print(p)
		temp, residuals, rank, sv = lstsq(homography, p, -1)
		#print("residuals are", residuals)
		x = temp[0]
		y = temp[1]
		z = temp[2]
		realPoints.append([(x/z), (y/z)])

	realPoints = np.array(realPoints)
	print("----test----, after homography, real point coordinate is \n")
	#print("----test----, after homography, real point coordinate is \n", file = open("output.txt", "a"))
	print(realPoints)
	#print(realPoints, file = open("output.txt", "a"))

	tri = Delaunay(realPoints)

	#print("----test----, calculateContourArea, tri.simplices is \n", tri.simplices)
	#print("----test----, points, tri.simplices is \n", realPoints[tri.simplices])
	#print("----test----,pixelPoints[:,1]", realPoints[:,1])
	plt.triplot(realPoints[:,0], realPoints[:,1], tri.simplices.copy())
	plt.plot(realPoints[:,0], realPoints[:,1], 'o')
	plt.savefig("report"+str(shapeIndex)+".png")
	plt.show()
	#plt.savefig("report"+str(shapeIndex)+".png")
	#shapeIndex = shapeIndex+1
	# Calculate the area
	totalArea = 0
	for triangle in realPoints[tri.simplices]:
		print("----test----,triangle is ", triangle)
		totalArea  += triArea(triangle)

	print("----test----, total area is ", totalArea )
	return totalArea




########################################################################
# The aim of this function is to test, whether homography is correct.
# Input pixel points is in the form pixelCoordinates[i] = [[x], [y], [1]].
# There will be an image showing the real coordinates. You can print to
# manually check
########################################################################
def correctnessTest(pixelPoints, homography, correntPoints):
	pixel3dPoints = []
	for point in  pixelPoints:
		pixel3dPoints.append([[point[0]], [point[1]],[1]])
	
	realPoints = []
	for point in pixel3dPoints:
		point = np.array(point)
		#print("in correctnessTest point is", point)
		temp, residuals, rank, sv = lstsq(homography, point, -1)
		#print("in correctnessTest calculated real points is")
		#print(temp)
		x = temp[0][0]
		y = temp[1][0]
		z = temp[2][0]
		#print("in correctnessTest after divide by z is")
		#print(int(x/z), int(y/z))
		realPoints.append((round(x/z,2), round(y/z, 2)))
	#print("finally, the caculated points coordinates are")
	#print(realPoints)
	temp = np.zeros([50, 50,3], dtype = np.uint8)
	temp.fill(255)
	#这里出现的图形，和我平时纸上画的是上下颠倒的，但是没有关系，只要形状对了就可以
	for point in realPoints:
		cv.circle(temp, (int(point[0]),int(point[1])), 1, (0,0,255), -1 )
	cv.imshow('temp', temp)
	cv.waitKey(0)			
	cv.destroyAllWindows()
	print("correct value \t", "calculated value \t", "error \t\t")

	totalError = 0
	totalDistanecError = 0
	for i in range(len(realPoints)):
		correct = correntPoints[i]
		calculated = realPoints[i]
		euclidean = euclideanDistance(correct, calculated)
		#x_error = (calculated[0] - correct[0])  / correct[0]
		#y_error = (calculated[1] - correct[1])  / correct[1]
		#totalError = totalError + abs(x_error) + abs(y_error)
		totalDistanecError = totalDistanecError + euclidean
		# print(correct[0],"\t\t",calculated[0], "\t\t", x_error, "\t\t")
		# print(correct[1],"\t\t",calculated[1], "\t\t", y_error, "\t\t")
		print("(",correct[0],",",correct[1],")"," &\t",calculated, "&\t", round(euclidean,2), "\\\\ \t\t")
	print("on average, the bias between real coordinates and measured coordinates is", totalDistanecError/len(realPoints))
	print("the bias between real & measured is", totalDistanecError/len(realPoints), file = open("output.txt", "a"))

	# calcualte the error range

	return

########################################################################
# Given sortedCorner(pixel coordinates) and patternPoints(real coordinate)
# To calculate calibration matrix
########################################################################
def calculateCalibration(sortedCorners, patternPoints):
	camera_matrix = np.zeros((3, 3),'float32')
	height = img.shape[0]
	weight = img.shape[1]#3080
	focal = 18
	if(len(sys.argv) > 2):
		focal = int(sys.argv[2])
	# camera_matrix[0,0]= weight/focal
	# camera_matrix[1,1]= height/focal
	camera_matrix[0,0]= (focal/23.5)*weight
	camera_matrix[1,1]= (focal/15.6)*height	
	camera_matrix[2,2]= 1.0
	camera_matrix[0,2]= img.shape[1]/2
	camera_matrix[1,2]= img.shape[0]/2

	dist_coefs = np.zeros(0,'float32')
	patternPoints = np.array(patternPoints, 'float32') 
	sortedCorners = np.array(sortedCorners, 'float32') 
	rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(
		[patternPoints], [sortedCorners], 
		(img.shape[1],img.shape[0]),
		camera_matrix,dist_coefs,  
		flags=cv.CALIB_FIX_K3 + cv.CALIB_USE_INTRINSIC_GUESS)
	return rms, camera_matrix, dist_coefs, rvecs, tvecs


########################################################################
# Find 50 points on ruler, and cut all the other non-important thing
########################################################################
def findRulerCorners(corners):
	lines = []
	for i in range(0, len(corners)):
		for j in range(i, len(corners)):
			abandon = False
			pointsNo, fittedPoints = pointsFitting(corners, corners[i], corners[j])
			for point in fittedPoints:
				if cornerInLines(point, lines):
					abandon = True
			if (pointsNo == 10 and not abandon):
				print("----test----", pointsNo)
				print("----test----", fittedPoints)
				lines.append(fittedPoints)

	print("----test----", "length of line is")
	print(len(lines))

	print("----test----")
	for line in lines:
		img1 = img.copy()
		x1 = int(line[0][0])
		y1 = int(line[0][1])
		x2 = int(line[9][0])
		y2 = int(line[9][1])
		cv.line(img1, (x1, y1), (x2, y2), (0, 0, 225), 5)
		cv.namedWindow('img1', cv.WINDOW_NORMAL)
		cv.resizeWindow('img1', int(img.shape[1]/resizeAspect), int(img.shape[0]/resizeAspect))
		cv.imshow('img1', img1)
		cv.waitKey(0)
		cv.destroyAllWindows()
	if (len(lines) == 4):
		print("Successfully located 50 ruler points ...")
		return lines
	else:
		print("Error occurs when perform findRulerCorners ...")
		return lines

########################################################################
# Decide whether corner exists in lines
########################################################################
def cornerInLines(corner, lines):
	for line in lines:
		if corner in line:
			return True
	return False


########################################################################
# Test how many points are on the line AB among corners.
########################################################################
def pointsFitting(corners, A, B):
	count = 0
	fittedPoints = []
	Ax = A[0]
	Ay = A[1]
	Bx = B[0]
	By = B[1]
	for i in range(0, len(corners)):
		corner = corners[i]
		x = corner[0]
		y = corner[1]
		# if the slope is infinity
		if (Ax - Bx == 0 and almostEqual_Slope(x, Ax)):
			fittedPoints.append((x,y))
			count += 1
		# for AB with normal slope
		elif (Ax - Bx != 0 and almostEqual_Slope(y - Ay, (x - Ax) * ((By - Ay) / (Bx - Ax))) ):
			fittedPoints.append((x,y))
			count += 1
	return count, fittedPoints


########################################################################
# Instead of euqal, set a loose equal to compare whether A == B
########################################################################
def almostEqual_Slope(A, B):
	min_aspect = 0.95
	max_aspect = 1.05
	if(A >= B * min_aspect and A <= B * max_aspect):
		return True
	else:
		return False



def on_mouse4(event, x, y, flags, param):
	global ROI,min_y, min_x
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	if event == cv.EVENT_LBUTTONDOWN:
		cv.circle(ROI, (x, y), 2, (0,0,255),-1)
		cv.imshow('ROI',ROI)

		# ROI2= gray[  y - 5 :y + 5, x -5 : x + 5 ]
		# points = cv.goodFeaturesToTrack(ROI2, 1, 0.01, 10)
		# x2, y2 = points[0].ravel()
		# cv.circle(ROI, (int(x -5 + x2) , int(y - 5 + y2)), 2, (0,255,0),-1)
		# contourPoints.append((x -5 + x2, y - 5 + y2))
		contourPoints.append((x+min_x , y+min_y))
		print("corner detect", int(x+min_x ) , int(y+min_y))
		print("i clicked ",x+min_x ,y+min_y)
		#print("contourPoints is", contourPoints)


def on_mouse3(event, x, y, flags, param):
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	if event == cv.EVENT_LBUTTONDOWN:
		cv.circle(img, (x, y), 9, (0,0,255),-1)
		cv.imshow('img',img)

		ROI2= gray[y - 5 :y + 5, x -5 : x + 5]
		points = cv.goodFeaturesToTrack(ROI2, 1, 0.01, 10)
		x2, y2 = points[0].ravel()
		cv.circle(img, (int(x -5 + x2) , int(y - 5 + y2)), 5, (0,255,0),-1)
		contourPoints.append((x -5 + x2, y - 5 + y2))
		print("corner detect", int(x -5 + x2) , int(y - 5 + y2))
		print("i clicked ",x ,y)
		print("contourPoints is", contourPoints)


def on_mouse2(event, x, y, flags, param):
	global ix, iy, min_x, min_y, ROI

	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	if event == cv.EVENT_LBUTTONDOWN:
		cv.circle(ROI, (x,y), 5, (0,0,255),-1)
		ix, iy = min_x+x, min_y+y

		ROI2= gray[iy - 5 :iy + 5, ix -5 : ix + 5]
		# use Shi Tomasi corner detector
		points = cv.goodFeaturesToTrack(ROI2, 1, 0.01, 10)
		x2, y2 = points[0].ravel()
		cv.circle(ROI, (int(x2)+ix-5-min_x, int(y2)+iy-5-min_y), 5, (0,255,0),-1)
		print("corner detector get", x2+ix-5, y2+iy-5)
		print("i clicked ,",ix ,iy)

		manualCorners.append((int(x2)+ix-5, int(y2)+iy-5))
		print("in on_mouse2, the list is", manualCorners)


def on_mouse(event, x, y, flags, param):
	global img
	img2 = img.copy()
	global min_x, min_y, width, height, point1, point2
	global ROI
	if event == cv.EVENT_LBUTTONDOWN:
		point1 = (x,y)
		cv.circle(img2, point1, 10, (0,255,0), 5)
		cv.imshow('temp',img2)
	elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON):
		cv.rectangle(img2, point1, (x, y),(255, 0, 0), 5)
		cv.imshow('temp',img2)
	elif event == cv.EVENT_LBUTTONUP:
		point2 = (x,y)
		cv.rectangle(img2, point1, point2, (0,0,255), 5)
		cv.imshow('temp', img2)
		min_x = min(point1[0],point2[0])     
		min_y = min(point1[1],point2[1])
		width = abs(point1[0] - point2[0])
		height = abs(point1[1] - point2[1])

########################################################################
# For 4 lines, 10 points on one line. We need to arrange these 10 points
# in sequence, according sequenc.jpg
########################################################################
def sortLinesPoints(lines):
	# select 6 intersection points of 4 lines( 2 points should be infinity )
	lineVectors = []
	intersections = []
	distances = []
	for line in lines:
		vx, vy, cx, cy = cv.fitLine(np.float32(line), cv.DIST_L2, 0, 0.01, 0.01)
		lineVectors.append([(vx, vy), (cx, cy)])
		line_img = img.copy()
		cv.line(line_img, (int(cx-vx*4000),int(cy-vy*4000)), (int(cx+vx*4000), int(cy+vy*4000)), (0,0,255))
		cv.namedWindow('line_img', cv.WINDOW_NORMAL)
		cv.resizeWindow('line_img', int(img.shape[1]/resizeAspect), int(img.shape[0]/resizeAspect))
		cv.imshow('line_img', line_img)
		cv.waitKey(0)
	for i in range(0, len(lineVectors)):
		for j in range(i, len(lineVectors)):
			R = intersection(lineVectors[i], lineVectors[j])
			if R:
				if (not isRInfinity(R)):
					intersections.append((float(R[0]), float(R[1])))
					distances.append(calculateDistance(lines[i], lines[j], R))

	#step 2 计算交点与lines之间所有点的距离差，最大是D，最小是B
	maxDis = float(0)
	maxIndex = 0
	minDis = float("inf")
	minIndex = 0
	for i in range(0, len(distances)):
		if (float(distances[i][3]) > maxDis):
			maxDis = distances[i][3]
			maxIndex = i
		if (float(distances[i][3]) < minDis):
			minDis = distances[i][3]
			minIndex = i 
	D = distances[maxIndex][2]
	B = distances[minIndex][2]


	#step 3 分出lOne, lTwo, lThree, lFour, 注意，这里的XY轴和平时的不一样，所以cross正负对应是反的
	DB = np.array([float(B[0]-D[0]), float(B[1]-D[1]), 0])
	line1 = distances[maxIndex][0]
	vector1 = np.array([float(line1[0][0]-D[0]), float(line1[0][1]-D[1]), 0])
	crossProduct = np.cross(DB, vector1)

	if crossProduct[2] < 0:
		lineOne = distances[maxIndex][0]
		lineFour = distances[maxIndex][1]
	else:
		lineOne = distances[maxIndex][1]
		lineFour = distances[maxIndex][0]
	line3 = distances[minIndex][0]
	vector3 = np.array([float(line3[0][0]-B[0]), float(line3[0][1]-B[1]), 0])
	crossProduct = np.cross(DB, vector3)
	if crossProduct[2] < 0:
		lineTwo = distances[minIndex][0]
		lineThree = distances[minIndex][1]
	else:
		lineTwo = distances[minIndex][1]
		lineThree = distances[minIndex][0]


	# print("----test----", "sequence of four lines")
	# temp = img.copy()
	# cv.line(temp, lineOne[0], lineOne[3], (0,0,255))
	# cv.line(temp, lineTwo[0], lineTwo[3], (0,255,0))
	# cv.line(temp, lineThree[0], lineThree[3], (255,0,0))
	# cv.line(temp, lineFour[0], lineFour[3], (255,255,255))
	# cv.namedWindow('sequenceChecking', cv.WINDOW_NORMAL)
	# cv.resizeWindow('sequenceChecking', int(img.shape[1]/resizeAspect), int(img.shape[0]/resizeAspect))
	# cv.imshow('sequenceChecking', temp)
	# cv.waitKey(0)

	# step four 把四条线中的点，按照关系排列组合
	lines = [lineOne, lineTwo, lineThree, lineFour]
	pointsDistance = {}
	for i in range(0, 4):
		line = lines[i]
		for point in line:
			pointsDistance[point] = int(euclideanDistance(D, point))
		lines[i][:] = []
		for key, value in sorted(pointsDistance.items(), key = operator.itemgetter(1)):
			lines[i].append(key)
		pointsDistance.clear()
	temp = img.copy()


	return lines



########################################################################
# Given two lines A and B, to detect their intersection point
########################################################################
def intersection(A, B):
	Avx = A[0][0]
	Avy = A[0][1]
	Acx = A[1][0]
	Acy = A[1][1]
	Bvx = B[0][0]
	Bvy = B[0][1]
	Bcx = B[1][0]
	Bcy = B[1][1]
	A1 = (Acx-Avx*2000, Acy-Avy*2000)
	A2 = (Acx+Avx*2000, Acy+Avy*2000)
	B1 = (Bcx-Bvx*2000, Bcy-Bvy*2000)
	B2 = (Bcx+Bvx*2000, Bcy+Bvy*2000)

	# L1 = [A1[1] - A2[1], A2[0] - A1[0], -(A1[0]*A2[1] - A2[0]*A1[1])]
	# L2 = [B1[1] - B2[1], B2[0] - B1[0], -(B1[0]*B2[1] - B2[0]*B1[1])]

	# D  = L1[0] * L2[1] - L1[1] * L2[0]
	# Dx = L1[2] * L2[1] - L1[1] * L2[2]
	# Dy = L1[0] * L2[2] - L1[2] * L2[0]

	# if D != 0:
	# 	x = Dx / D
	# 	y = Dy / D
	# 	temp = img.copy()
	# 	cv.line(temp, A1, A2, (0,0,255))
	# 	cv.line(temp, B1, B2, (0,0,255))
	# 	cv.circle(temp, (x, y), 3, (0,0,255), -1 )
	# 	cv.namedWindow('temp', cv.WINDOW_NORMAL)
	# 	cv.resizeWindow('temp', int(img.shape[1]/resizeAspect), int(img.shape[0]/resizeAspect))
	# 	cv.imshow('temp', temp)
	# 	cv.waitKey(0)
	# 	return x,y
	# else:
	# 	return False

	N1 = Avy/Bvy - Avx/Bvx
	N2 = (Acx-Bcx)/Bvx - (Acy-Bcy)/Bvy

	M1 = Bvx/Avx - Bvy/Avy
	M2 = (Bcy-Acy)/Avy - (Bcx-Acx)/Avx
	if (N1 == 0):
		return False
	else:
		t = M2/M1
		x = t*Bvx+Bcx
		y = t*Bvy+Bcy
		return (x, y)

########################################################################
# Given point R, whether this is out of image shape
########################################################################
def isRInfinity(R):
	height = img.shape[0]
	width = img.shape[1]
	if (int(R[0]) > width or int(R[1]) > height or R[0] < 0 or R[1]<0):
		return True
	return False

########################################################################
# Given line A and line B, their intersection R, return distance between
# R and line points. (and these two lines)
########################################################################
def calculateDistance(A, B, R):
	dis = 0
	for point in A:
		dis += math.sqrt((point[0] - R[0])**2 + (point[1] - R[1])**2)
	for point in B:
		dis += math.sqrt((point[0] - R[0])**2 + (point[1] - R[1])**2)
	return A, B, R, dis


########################################################################
# Get euclidean Distance between A and B
########################################################################
def euclideanDistance(pointA, pointB):
	dist = math.sqrt((pointB[0] - pointA[0])**2 + (pointB[1] - pointA[1])**2)
	return dist 


########################################################################
# Calculate the area given three coordinates
# input is like 
# [[72.18761378 59.06871943]
# [37.96444783 35.71516127]
# [95.18101714 13.02890558]]
########################################################################
def triArea(triangle):
	x1 = triangle[0][0]
	y1 = triangle[0][1]
	x2 = triangle[1][0]
	y2 = triangle[1][1]
	x3 = triangle[2][0]
	y3 = triangle[2][1]

	area = ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)) / 2
	return area


########################################################################
# Get the pixels by mouse clicking and save to a vector
########################################################################
# def getPixelCoordinateListByMouse():
# 	global min_x, min_y, width, height
# 	global ROI, manualCorners
# 	if( len(sys.argv) >2 and sys.argv[2] == "points"):

# 		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 		gray = cv.GaussianBlur(gray, (5,5),0)

# 		# select region of interest, so that corner detection can be more accurate
# 		cv.namedWindow('img', cv.WINDOW_NORMAL)
# 		cv.resizeWindow('img', int(img.shape[1]/resizeAspect), int(img.shape[0]/resizeAspect))
# 		cv.setMouseCallback('img', on_mouse)
# 		cv.imshow('img', img)
# 		cv.waitKey(0)
# 		print("get region of interest ...")


# 		ROI = img[min_y:min_y+height, min_x:min_x+width]
# 		cv.namedWindow('ROI', cv.WINDOW_NORMAL)
# 		cv.resizeWindow('ROI', int(height/resizeAspect), int(width/resizeAspect))
# 		cv.setMouseCallback('ROI', on_mouse2)
# 		cv.imshow('ROI', ROI)
# 		while(1):
# 			cv.imshow('ROI', ROI)
# 			k = cv.waitKey(20)&0xFF
# 			if k==27:
# 				break
# 			elif k == ord('d'):
# 				del manualCorners[len(manualCorners)-1]
# 				print ("delete one point, len is now,")
# 				print(manualCorners)
# 		cv.destroyAllWindows()
# 	else:
# 		#manualCorners = [(1859, 1019), (1866, 925), (1868, 839), (1874, 753), (1874, 670), (1879, 590), (1881, 512), (1883, 439), (1884, 364), (1890, 296), (1961, 1022), (1961, 936), (1965, 846), (1967, 761), (1967, 680), (1970, 603), (1967, 523), (1970, 446), (1971, 374), (1974, 305), (2006, 1180), (2092, 1189), (2189, 1197), (2279, 1206), (2371, 1211), (2460, 1220), (2551, 1227), (2638, 1231), (2726, 1238), (2815, 1244), (2000, 1275), (2093, 1282), (2187, 1288), (2281, 1294), (2375, 1302), (2468, 1309), (2557, 1315), (2645, 1322), (2735, 1331), (2823, 1334)]
# 		#the following is for front.JPG
# 		#manualCorners = [(1862, 1014), (1866, 924), (1869, 836), (1872, 752), (1875, 669), (1878, 589), (1881, 512), (1883, 440), (1884, 364), (1886, 294), (1957, 1023), (1960, 934), (1962, 846), (1964, 762), (1965, 679), (1966, 601), (1966, 521), (1967, 448), (1969, 373), (1970, 303), (2004, 1180), (2090, 1187), (2187, 1196), (2277, 1203), (2372, 1212), (2459, 1219), (2551, 1226), (2636, 1232), (2728, 1240), (2812, 1246), (2002, 1273), (2091, 1279), (2188, 1287), (2279, 1293), (2375, 1302), (2463, 1308), (2557, 1315), (2643, 1321), (2735, 1329), (2820, 1334)]


# 		# the following is for 2.JPG
# 		manualCorners = [(1572, 997), (1592, 954), (1613, 910), (1632, 870), (1651, 831), (1668, 794), (1686, 759), (1700, 728), (1716, 693), (1729, 663), (1659, 1007), (1678, 963), (1697, 919), (1713, 879), (1730, 839), (1745, 802), (1760, 766), (1773, 734), (1788, 699), (1800, 669), (1679, 1116), (1767, 1127), (1864, 1139), (1956, 1150), (2055, 1162), (2149, 1173), (2249, 1185), (2345, 1195), (2450, 1206), (2548, 1216), (1660, 1167), (1752, 1178), (1851, 1190), (1945, 1201), (2046, 1214), (2143, 1225), (2247, 1237), (2345, 1249), (2452, 1260), (2553, 1271)]

# 		# the following is for 1.jpg
# 		#manualCorners=[(258, 884), (256, 825), (254, 761), (252, 704), (250, 641), (247, 585), (243, 523), (241, 467), (239, 407), (237, 352), (318, 880), (316, 821), (313, 757), (310, 701), (308, 637), (304, 582), (301, 520), (299, 469), (295, 404), (292, 350), (364, 965), (422, 961), (485, 956), (538, 954), (605, 948), (663, 944), (724, 941), (782, 938), (837, 934), (900, 930), (367, 1025), (425, 1021), (489, 1015), (546, 1011), (609, 1007), (668, 1004), (730, 1000), (787, 995), (849, 990), (905, 987)]

# 	return manualCorners

########################################################################
# This fucntion pass a new image temp, therefore, it get the pixel 
# coordinate of temp instead of 'img'
########################################################################
def getPixelCoordinateListByMouse(temp, times):
	global min_x, min_y, width, height
	global ROI, manualCorners
	manualCorners.clear()
	# gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
	# gray = cv.GaussianBlur(gray, (5,5),0)

	# select region of interest, so that corner detection can be more accurate
	for i in range(times):
		cv.namedWindow('temp', cv.WINDOW_NORMAL)
		cv.resizeWindow('temp', int(temp.shape[1]/resizeAspect), int(temp.shape[0]/resizeAspect))
		cv.setMouseCallback('temp', on_mouse)
		cv.imshow('temp', temp)
		cv.waitKey(0)
		print("get region of interest ...")


		ROI = temp.copy()[min_y:min_y+height, min_x:min_x+width]
		cv.namedWindow('ROI', cv.WINDOW_NORMAL)
		cv.resizeWindow('ROI', int(height/resizeAspect), int(width/resizeAspect))
		cv.setMouseCallback('ROI', on_mouse2)
		cv.imshow('ROI', ROI)
		while(1):
			cv.imshow('ROI', ROI)
			k = cv.waitKey(20)&0xFF
			if k==27:
				break
			elif k == ord('d'):
				del manualCorners[len(manualCorners)-1]
				print ("delete one point, len is now,")
				print(manualCorners)
		cv.destroyAllWindows()
	return manualCorners

########################################################################
# Test the area calculation using homography, delauny
########################################################################
def testAreaCalculation(homography):
	global shapeIndex
	shapeIndex=20
	output=[]
	totalError=0
	continue1 = 1
	while(continue1):
		print("Get countours from image ...")
		targetPoints = findContourPoints()
		print("Successfully get the countours coordinates, shown as bellow:")
		print(targetPoints)


		print("Calculate the countour area ...")
		area = calculateContourArea(targetPoints, homography)
		shapeIndex=shapeIndex+1
		print("Successfully get the area, shown as bellow")
		print("area is", area)

		print("Comparing the calculated measuring plane area with real area...")
		real = input("please input the real area...")
		error =100 * (area - float(real))/float(real)
		totalError =totalError + abs(error)
		output.append([real, area, error])
		print("error is",error,"%")
		continue1 = int(input("continue? if yes, enter 1. Remember to screen shot!!! \n"))

	print("imagename\t","real\t","calculated\t\t","error\t")
	print("totalError is", totalError)
	for result in output:
		print(sys.argv[1],"&\t",round(float(result[0]),2),"&\t", round(float(result[1]),2),"&\t", round(float(result[2]),2),"\%\\\\\t")
		print(sys.argv[1],"&\t",round(float(result[0]),2),"&\t", round(float(result[1]),2),"&\t", round(float(result[2]),2),"\%\\\\\t", file = open("output.txt", "a"))
	print("average error is", totalError/len(output))

	
########################################################################
# Transfer the image into calibrated ones.
########################################################################
def undistortImage(image, mtx, dist):
	pixelImg = image
	h, w = image.shape[:2]

	newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h),1,(w,h))	
	dst = cv.undistort(pixelImg, mtx, dist, None, newcameramtx)

	cv.namedWindow('calibrated', cv.WINDOW_NORMAL)
	cv.resizeWindow('calibrated', int(image.shape[1]/resizeAspect), int(image.shape[0]/resizeAspect))
	cv.imshow('calibrated', dst)
	cv.waitKey(0)


	return calibrated


def changeImg(img, homography):
	global shapeIndex
	height, weight = img.shape[:2]
	result = np.zeros([3000, 3000,3], dtype = np.uint8)
	result.fill(255)
	cv.namedWindow('temp', cv.WINDOW_NORMAL)
	cv.resizeWindow('temp', int(img.shape[1]/resizeAspect), int(img.shape[0]/resizeAspect))
	cv.imshow('temp', img)
	cv.waitKey(0)
	cv.destroyAllWindows()

	#这里出现的图形，和我平时纸上画的是上下颠倒的，但是没有关系，只要形状对了就可以
	for x in range(height):
		for y in range(weight):
			color = img[x,y]
			#print("color is ", color)
			point = [[y],[x],[1]]
			point = np.array(point)
			temp, residuals, rank, sv = lstsq(homography, point, -1)
			#print("accessing point", x, y)
			x1 = temp[0][0]
			y1 = temp[1][0]
			z1 = temp[2][0]
			
			realPoint = [1000+int(y1/z1), 1000+int(x1/z1)]
			#print(realPoint[0],realPoint[1])
			# if (x1/z1) >0 and (y1/z1) >0:
			# 	result[realPoint[0],realPoint[1]] = color
			result[realPoint[0],realPoint[1]] = color
	cv.namedWindow('temp', cv.WINDOW_NORMAL)
	cv.resizeWindow('temp', int(3000/resizeAspect), int(3000/resizeAspect))
	cv.imshow('temp', result)
	cv.waitKey(0)
	cv.destroyAllWindows()
	cv.imwrite("result1.jpg",result)
	
	return



def main():

	global img
	times=1
	img = cv.imread(sys.argv[1])
	print("Get pixel coordinates of corners...")
	#pixelCoordinates = getPixelCoordinateList()
	pixelCoordinates = getPixelCoordinateListByMouse(img, times)
	print("Successfully get pixel coordinates, shown as bellow: ")
	print(pixelCoordinates)
	print("length is", len(pixelCoordinates))



	print("Get real coordinates of corners ...")
	realCoordinates = getRealCoordinateList(times)
	print("Successfully get real coordinates, shown as bellow:")
	print(realCoordinates)

	print("Calculate homography matrix ...")
	homography =  calculatehomographyMatrix(pixelCoordinates, realCoordinates)
	print("Successfully get homography matrix, shown as bellow:")
	print(homography)
	
	# print("Use homography to change the whole img")
	# changeImg(img, homography)



	print("Test the correctness of homography ...")
	correctnessTest(pixelCoordinates, homography,realCoordinates)


	print("Test the correctness of area ...")
	testAreaCalculation(homography)



if __name__ == '__main__':
    main()
