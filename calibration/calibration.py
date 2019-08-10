# -*- coding: utf-8 -*-
import cv2 as cv 
import numpy as np 
import sys
import math
import operator


img_color = cv.imread(sys.argv[1])


def GetQuadrangleHypothese(contours, hierarchy, quads,class_id, img):
	index = 0
	min_box_size = float(10.0)
	min_aspect_ratio = float(0.3)
	max_aspect_ratio = float(3.0)
	#pattern_size =(9,6) #this represent 几乘以几 的表盘
	for contour in contours:
		# if (hierarchy[index][3] != -1):
		# 	continue; #skip holes
		rect = cv.minAreaRect(contour)

		#from documentation, https://docs.opencv.org/3.1.0/db/dd6/classcv_1_1RotatedRect.html
		#the return sequence is center, size, angle
		angle = rect[2]
		center_x = rect[0][0]
		center_y = rect[0][1]
		width = rect[1][0]
		height = rect[1][1]
		box_size = max(width, height)
		if( box_size < min_box_size or box_size > min(img.shape[1],img.shape[0])/9):
			continue

		aspect_ratio =  width / max(height, 1)
		if( aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio):
			continue
		#print(rect)

		box = cv.boxPoints(rect)
		box = np.int0(box)
		#print("for four coordinates,", box)
		cv.drawContours(img,[box],0,(0,0,255),10)
		quads.append((box_size, class_id,rect))
		index=index+1

	cv.namedWindow('checking rec', cv.WINDOW_NORMAL)
	cv.resizeWindow('checking rec', int(img.shape[1]/4), int(img.shape[0]/4))
	cv.imshow('checking rec', img)
	cv.waitKey(0)
	cv.destroyAllWindows()

	return

def checkQuads(quads):
	min_quads_count = 5
	quads.sort()
	#print(quads)
	size_rel_dev = float(0.4)
	for i in range(0, len(quads)):
		for j in range(i+1, len(quads)):
			if( quads[j][0]/ quads[i][0] > 1 + size_rel_dev):
				break
		if( j + 1 > min_quads_count + i):
			#check the number of black and white squares
			black_count = 10
			white_count = 10
			counted_b, counted_w = countClasses(quads, i, j)
			if( counted_b < black_count*0.8 or counted_w < white_count*0.8):
				continue
			#可以在if里面加上 counted_b > black_count * 1.2, counted_w > white_count*1.2,来限制上限
			return True
	return False

def countClasses(quads, idx1, idx2):
	black_count = 0
	white_count = 0
	for i in range(idx1, idx2):
		if(quads[i][1] == 0):
			black_count+=1
		else:
			white_count+=1
	return white_count, black_count

def calculateDistance(pointA, pointB):
	dist = math.sqrt((pointB[0] - pointA[0])**2 + (pointB[1] - pointA[1])**2)
	return dist 

def almostEqual(a, b):
	min_aspect = 0.85
	max_aspect = 1.15
	#print("comparing ",a,"with",b*min_aspect,"and",b*max_aspect)
	if( a >= b*min_aspect and a <= b*max_aspect):
		return True
	else:
		#print("they are not almost equal")
		return False

def angleEqual(a, b):
	min_side = -10
	max_side = 10
	if( a >= b+min_side and a <= b+max_side):
		return True
	else:
		return False

def deleteQuadsBySize(quads, box_sizes, angles):
	#find the most frequent size
	targetSizeStart = 0
	targetSize = 0
	box_sizes.sort()
	print(box_sizes)

	temp = 0
	max_count = 0
	for i in range(1,len(box_sizes)):
		#print("compare",box_sizes[i],"and",box_sizes[i-1],temp ,"? larger than",max_count)
		if( almostEqual(box_sizes[i],box_sizes[i-1])):
			if( temp == 0):
				 temp2= box_sizes[i-1]
			temp += 1
		else:
			if( temp > max_count):
				targetSizeStart = temp2
				max_count = temp
				targetSize = box_sizes[i-1]
			temp = 0
	if(almostEqual(box_sizes[len(box_sizes)-1],box_sizes[len(box_sizes)-2]) and temp > max_count):
		targetSizeStart = temp2
		max_count = temp
		targetSize = box_sizes[len(box_sizes)-1]
	#print("for start and end of most frequent",targetSizeStart,targetSize)
	targetSize = (targetSizeStart+targetSize)/2
	print("most frequent is ",targetSize,"quads number is ",len(quads))
	#delete the unqualified quads
	deletes=[]
	for i in range(0,len(quads)):
		quad = quads[i]
		size = quad[0]
		#print("the size is ",size)
		if( not almostEqual(size, targetSize)):
			deletes.append(i)
	#print(deletes)
	for i in reversed(deletes):
		del quads[i]
		del angles[i]

	for quad in quads:
		rect = quad[2]
		box = cv.boxPoints(rect)
		box = np.int0(box)
		#print("for four coordinates,", box)
		cv.drawContours(img_color,[box],0,(0,0,255),10)
	cv.namedWindow('deletebySize', cv.WINDOW_NORMAL)
	cv.resizeWindow('deletebySize', int(img.shape[1]/4), int(img.shape[0]/4))
	cv.imshow('deletebySize', img_color)
	return quads, angles

def deleteQuadsByAngle(quads, angles):
	#keep cut un-neccessary quad according to angle
	targetAngleStart = 0
	targetAngle = 0
	angles.sort()
	print(angles)

	temp = 0
	temp2 = 0
	max_count = 0
	for i in range(1,len(angles)):
		#print("Angle compare",angles[i],"and",angles[i-1],temp ,"? larger than",max_count)
		if( angleEqual(angles[i],angles[i-1])):
			if( temp == 0):
				 temp2= angles[i-1]
			temp += 1
		else:
			if( temp > max_count):
				targetAngleStart = temp2
				max_count = temp
				targetAngle = angles[i-1]
			temp = 0
	if(angleEqual(angles[len(angles)-1],angles[len(angles)-2]) and temp > max_count):
		targetAngleStart = temp2
		max_count = temp
		targetAngle = angles[len(angles)-1]
	#print("for start and end of most frequent",targetAngleStart,targetAngle)
	targetAngle = (targetAngleStart+targetAngle)/2
	#print("most frequent is ",targetAngle,"quads number is ",len(quads))
	#using angle to delete the unqualified quads
	deletes=[]
	for i in range(0,len(quads)):
		quad = quads[i]
		angle = quad[2][2]
		#print("the angle is ",angle)
		if( not angleEqual(angle, targetAngle)):
			deletes.append(i)
	#print(deletes)
	for i in reversed(deletes):
		del quads[i]
	return quads

def slopeEqual(numberA, numberB):
	min_aspect = 0.85
	max_aspect = 1.15
	if(numberA >= numberB * min_aspect and numberA <= numberB * max_aspect):
		return True
	else:
		return False



def pointsFittingLine(centers, pointA, pointB):
	count = 0
	indexes = []
	if(pointB[0] - pointA[0] == 0):
		for i in range(0, len(centers)):
			center = centers[i]
			if(slopeEqual(center[0],pointA[0])):
				count +=1
				indexes.append(i)
		return count, indexes
	else:
		slope = (pointB[1] - pointA[1])/(pointB[0] - pointA[0])
		for i in range(0, len(centers)):
			center = centers[i]
			if(slopeEqual(center[1]-pointA[1], slope*(center[0]-pointA[0]))):
				count += 1
				indexes.append(i)
		return count, indexes


def clusterPoints(centers):
	print(centers)
	clusters = []
	for i in range(0, len(centers)):
		for j in range(i, len(centers)):
			pointsNumber, clusteredIndexes = pointsFittingLine(centers, centers[i], centers[j])
			if(pointsNumber == 5):
				if(len(clusters) == 0):
					clusters.append(clusteredIndexes)
				elif(clusteredIndexes[0] != clusters[0][0]):
					clusters.append(clusteredIndexes)
					print("before clusterPoints return", clusters)
					return clusters
	return clusters

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def clusterSequence(centers, clusters):
	clustersA = clusters[0]
	clustersB = clusters[1]
	pointsA = []
	pointsB = []
	for i in clustersA:
		center = centers[i] 
		pointsA.append(center)
	for i in clustersB:
		center = centers[i]
		pointsB.append(center)
	#print("points A set has ", pointsA)
	#print("points B set has ", pointsB)
	#find the intersection of two lines
	L1 = line(pointsA[0], pointsA[3])
	L2 = line(pointsB[0], pointsB[3])
	R = intersection(L1, L2)
	if not R:
		print("No intersection detected")
	else:
		print("found intersection", R)
	#use cross product to define which axis is x(B), which is y(A)
	vectorA = np.array([pointsA[0][0]-R[0], pointsA[0][1]-R[1], 0])
	vectorB = np.array([pointsB[0][0]-R[0], pointsB[0][1]-R[1], 0])
	temp = np.cross(vectorA, vectorB)
	#print("cross product is ",temp[2])
	if temp[2] > 0:
		return clustersB, clustersA, R
	else:
		return clustersA, clustersB, R

def sortBox(box, R, center):
	sortedBox = []
	right = []
	left = []
	for point in box:
		vectorA = np.array([point[0]-R[0], point[1]-R[1], 0])
		vectorB = np.array([center[0]-R[0], center[1]-R[1], 0])
		temp = np.cross(vectorA, vectorB)
		if (temp[2]>0):
			right.append(point)
		else:
			left.append(point)
	if (len(right) != 2 or len(left) != 2):
		print("sortBox is wrong")
		return
	dist1 = calculateDistance(left[0], R)
	dist2 = calculateDistance(left[1], R)
	if (dist1 > dist2):
		sortedBox.append((left[1][0],left[1][1]))
		sortedBox.append((left[0][0],left[0][1]))
	else:
		sortedBox.append((left[0][0],left[0][1]))
		sortedBox.append((left[1][0],left[1][1]))	

	dist1 = calculateDistance(right[0], R)
	dist2 = calculateDistance(right[1], R)
	if (dist1 > dist2):
		sortedBox.append((right[0][0],right[0][1]))
		sortedBox.append((right[1][0],right[1][1]))
	else:
		sortedBox.append((right[1][0],right[1][1]))
		sortedBox.append((right[0][0],right[0][1]))

	return sortedBox

def subpix(img, quads):
	box_sizes = []
	class_ids = []
	distances = []
	angles = []
	#rects = []
	centerPoints = []
	for quad in quads:
		box_sizes.append(quad[0])
		class_ids.append(quad[1])
		#angle.append(quad[2][2])
		center_x = quad[2][0][0]
		center_y = quad[2][0][1]
		angles.append( quad[2][2])
		centerPoints.append((center_x,center_y))
		distances.append(calculateDistance((center_x, center_y), centerPoints[0]))
	
	quads, angles = deleteQuadsBySize(quads, box_sizes, angles)
	quads = deleteQuadsByAngle(quads, angles)

	if (len(quads) != 10):
		img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
		for quad in quads:
			rect = quad[2]
			box = cv.boxPoints(rect)
			box = np.int0(box)
			#print("for four coordinates,", box)
			cv.drawContours(img_color,[box],0,(0,0,255),10)
		cv.namedWindow('after_reduction', cv.WINDOW_NORMAL)
		cv.resizeWindow('after_reduction', int(img.shape[1]/4), int(img.shape[0]/4))
		cv.imshow('after_reduction', img_color)
		print("function deleteQuadsByAngle and  deleteQuadsBySize is wrong, only detects", len(quads))
		return
	#draw all quads after reduction
	img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
	for quad in quads:
		rect = quad[2]
		box = cv.boxPoints(rect)
		box = np.int0(box)
		#print("for four coordinates,", box)
		cv.drawContours(img_color,[box],0,(0,0,255),10)
	cv.namedWindow('after_reduction', cv.WINDOW_NORMAL)
	cv.resizeWindow('after_reduction', int(img.shape[1]/4), int(img.shape[0]/4))
	cv.imshow('after_reduction', img_color)

	#use line fitting to cluster into two parts
	line_img = img.copy()
	line_img[:]=255
	centers=[]
	for quad in quads:
		center_x = quad[2][0][0]
		center_y = quad[2][0][1]
		centers.append((center_x,center_y))
		cv.circle(line_img,(int(center_x),int(center_y)), 10, 0, -1)
																# minLineLength = 10
																# maxLineGap = 0
																# lines = cv.HoughLinesP(line_img, 1, np.pi/180, 100,minLineLength,maxLineGap)
																# for x1,y1,x2,y2 in lines[0]:
																#     cv.line(line_img,(x1,y1),(x2,y2),0,2)
																		# vx, vy, cx, cy = cv.fitLine(np.float32(centers), cv.DIST_L2, 0, 0.01, 0.01)
																		# cv.line(line_img, (int(cx-vx*400),int(cy-vy*400)), (int(cx+vx*100), int(cy+vy*100)), (0,0,255))
	print("!!!!!!!!!!!!!!!!!!!!", len(centers), len(quads))
	count = 0
	clusters = clusterPoints(centers)
	if(len(clusters) != 2):
		print("function clusterPoints clustered wrongly",clusters)
		#showClusteredPoints(clusters)
		return
	if( len(clusters[0]) != 5 or len(clusters[1]) != 5):
		print("function clusterPoints clustered wrongly, length of each is not 10")
		return

	#check the correct sequence of x axis and y axis
	clustersA, clustersB, intersection = clusterSequence(centers, clusters)
	centersAinSequence = {}
	centersBinSequence = {}
	for i in clustersA:
		centersAinSequence[i] = calculateDistance(intersection, centers[i])
	for i in clustersB:
		centersBinSequence[i] = calculateDistance(intersection, centers[i])
	#print("A set with respect to R",centersAinSequence)
	#print("B set with respect to R",centersBinSequence)
	sortedA = sorted(centersAinSequence.items(), key = operator.itemgetter(1))
	sortedB = sorted(centersBinSequence.items(), key = operator.itemgetter(1))
	#print("after sorting", sortedA, sortedB)
	
	#get the sorted quad 
	sortedQuads = []
	for item in sortedA:
		index = item[0]
		sortedQuads.append(quads[index])
	for item in sortedB:
		index = item[0]
		sortedQuads.append(quads[index])

	print("number of all quads is", len(sortedQuads))


	#get the rectangle corners
	Corners = []
	for quad in sortedQuads:
		rect = quad[2]
		box = cv.boxPoints(rect)
		box = np.int0(box)
		center_x = rect[0][0]
		center_y = rect[0][1]
		Corners.append((box, (center_x, center_y)))
	#print(sortedCorners)
	# sortedCorners
	print("number of corners", len(sortedQuads))
	sortedCorners = []
	for corner in Corners:
		box = corner[0]
		center = corner[1]
		#print("each rectangle's ends are", box)
		#sort the box into bottom left, up left, up right, bottom right. By comparing with the intersection point.
		box = sortBox(box, intersection, center)
		sortedCorners.append([box[0][0], box[0][1]])
		sortedCorners.append([box[1][0], box[1][1]])
		sortedCorners.append([box[2][0],box[2][1]])
		sortedCorners.append([box[3][0],box[2][1]])
	#print(sortedCorners)
	print("number of sorted corners are ", len(sortedCorners))

	patternPoints = []
	#correspond each image coordinate with real coordinate
	for i in range(0,5):
#王贾瑞
		# for j in range(0,2):
		# 	patternPoints.append([-5, 20*(1+i), 0])
		# 	patternPoints.append([-5, 20*(1+i)+10, 0])
		# 	patternPoints.append([5, 20*(1+i)+10, 0])
		# 	patternPoints.append([5, 20*(1+i), 0])
#王贾瑞，ps，还原的时候下面四行得去掉
		patternPoints.append([-5, 20*(1+i), 0])
		patternPoints.append([-5, 20*(1+i)+10, 0])
		patternPoints.append([5, 20*(1+i)+10, 0])
		patternPoints.append([5, 20*(1+i), 0])

	for i in range(0,5):
#王贾瑞
		# for j in range(0,2):
		# 	patternPoints.append([10+20*i, 5, 0])
		# 	patternPoints.append([10+20*i+10, 5, 0])
		# 	patternPoints.append([10+20*i+10, -5, 0])
		# 	patternPoints.append([10+20*i, -5, 0])
#王贾瑞，ps，还原的时候下面四行得去掉
		patternPoints.append([10+20*i, 5, 0])
		patternPoints.append([10+20*i+10, 5, 0])
		patternPoints.append([10+20*i+10, -5, 0])
		patternPoints.append([10+20*i, -5, 0])

	#print(patternPoints)
	cv.namedWindow('line_img', cv.WINDOW_NORMAL)
	cv.resizeWindow('line_img', int(img.shape[1]/4), int(img.shape[0]/4))
	cv.imshow('line_img', line_img)
	camera_matrix = np.zeros((3, 3),'float32')
	height = img.shape[0]
	weight = img.shape[1]
	focal = 19
	camera_matrix[0,0]= weight/focal
	camera_matrix[1,1]= height/focal
	camera_matrix[2,2]= 1.0
	camera_matrix[0,2]= img.shape[1]/2
	camera_matrix[1,2]= img.shape[0]/2

	dist_coefs = np.zeros(0,'float32')
	patternPoints = np.array(patternPoints, 'float32') 
	sortedCorners = np.array(sortedCorners, 'float32') 
	rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera([patternPoints], [sortedCorners], (img.shape[1],img.shape[0]),camera_matrix,dist_coefs,flags=cv.CALIB_USE_INTRINSIC_GUESS)

	print("\nRMS:", rms)
	print("camera matrix:\n", camera_matrix)
	print("distortion coefficients: ", dist_coefs.ravel())
	print("rotation vector: ", rvecs)
	print("translation vector: ", tvecs)


	cv.imshow('line_img', line_img)
	cv.waitKey(0)
	cv.destroyAllWindows()
	

	return


if __name__ == '__main__':
	img = cv.imread(sys.argv[1])
	# dim = (int(img.shape[1]/2),int(img.shape[0]/2))
	# img = cv.resize(img, dim)
	img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	#find chess board using fast, binary check
	blur = cv.GaussianBlur(img_gray, (5,5),0)
	ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

	white = th.copy()
	black = th.copy()

#从这里开始到王贾瑞结束，是要comment掉的
	quads=[]
	#fillQuads(white, black, 128, 128, quads)
	white_thresh = 128
	black_thresh = 128
	result = 0
	ret, thresh = cv.threshold(white, white_thresh, 255, cv.THRESH_BINARY)
	cv.namedWindow('thresh', cv.WINDOW_NORMAL)
	cv.resizeWindow('threshsh', int(img.shape[1]/4), int(img.shape[0]/4))
	cv.imshow('thresh', thresh)
	contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
	GetQuadrangleHypothese(contours, hierarchy, quads, 1, img)

#王贾瑞
	# ret, thresh = cv.threshold(black, black_thresh, 255, cv.THRESH_BINARY)
	# contours2, hierarchy2 = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
	# GetQuadrangleHypothese(contours2, hierarchy2, quads, 0, img)
#王贾瑞

	quads.sort()
#王贾瑞

	# for erosion_count in range(0,3):
	# 	#print("erosion_count is ", erosion_count)
	# 	if( result == 1):
	# 		break

	# 	if(erosion_count != 0):
	# 		#print("erosion for ",erosion_count," times")
	# 		white = cv.erode(white, (-1,-1), 1) #kernel: (-1,-1), iteration times:1
	# 		black = cv.dilate(black, (-1,-1), 1)

	# 	quads=[]
	# 	#fillQuads(white, black, 128, 128, quads)
	# 	white_thresh = 128
	# 	black_thresh = 128

	# #fill quads
	# 	ret, thresh = cv.threshold(white, white_thresh, 255, cv.THRESH_BINARY)
	# 	contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
	# 	GetQuadrangleHypothese(contours, hierarchy, quads, 1, img)
		

	# 	ret, thresh = cv.threshold(black, black_thresh, 255, cv.THRESH_BINARY)
	# 	contours2, hierarchy2 = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
	# 	GetQuadrangleHypothese(contours2, hierarchy2, quads, 0, img)
	# 	quads.sort()
		#print(quads)
	#check quads, check if there are many hypotheses with similar sizes
		# if(checkQuads( quads )):
		# 	result=1

	if(checkQuads( quads )):
		result=1	
	print("This image ",result," for calibration usage")
	subpix(img_gray, quads)


	# #testing
	# cv.drawContours(img, contours2, -1, (0,255,0), 3)
	# img2 = img.copy()
	# cv.drawContours(img2, contours, -1, (0,255,0), 3)

	# cv.imshow('white', img2)
	# cv.imshow('black', img)
	# cv.imshow('gray', img_gray)
	cv.waitKey(0)
	cv.destroyAllWindows()


		

	# print(img.shape)
	# print(img.dtype)
