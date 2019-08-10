import cv2.structured_light as sl
import cv2
import numpy as np

# params
width_projector = 1024 # default value
height_projector = 768 # default value

pattern_images = []

white_thresh = 127
black_thresh = 127

image_files = ''


def stereo_recitify():
    print("Rectifying images...")
    pass


def main():
    # Set up GraycodePattern with params
    graycode = sl.GrayCodePattern_create(width_projector, height_projector)

    # setting white_thresh and black_thresh value
    graycode.setWhiteThreshold(white_thresh)
    graycode.setBlackThreshold(black_thresh)

    number_pattern_images = graycode.getNumberOfPatternImages()

    # Loading calibration parameters
    # Stereo rectify
    # Loading pattern images
    # Loading images (all white + all black) needed for shadows computation


    # Storage for pattern

    # Generate the all-white and all-black images needed for shadows mask computation
    # Setting pattern window on second monitor (the projector's one)

    # Decode
    print("Decoding pattern...")
    retval, disparity_map = graycode.decode(pattern_images, flags=sl.DECODE_3D_UNDERWORLD)
    print("Pattern decoded")

    # Compute the point cloud
    disparity_map = np.float32(disparity_map)
    point_cloud = cv2.reprojectImageTo3D(disparity_map)

    # Compute a mask to remove background
    thresholded_disp = cv2.threshold(disparity_map, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    dst = cv2.resize(thresholded_disp, (640, 480))
    cv2.imshow("threshold disp otsu", dst)

    # Apply the mask to the point cloud
    # point_cloud_thresh = cv2.bitwise_and(point_cloud_thresh, thres)

    # Show the point cloud on viz


if __name__ == '__main__':
    main()
