from Utils.FileUtils import loadRectifyMap, getFramesImages
from Utils.ImageUtils import splitMergedImage
import cv2

left_map, right_map = loadRectifyMap()

images = getFramesImages()

counter = 1

stereo = cv2.StereoBM_create()

stereo.setMinDisparity(4)
stereo.setNumDisparities(128)
stereo.setBlockSize(21)
stereo.setSpeckleRange(16)
stereo.setSpeckleWindowSize(45)

for image in images:
    imgL, imgR = splitMergedImage(image)

    imgL = cv2.remap(imgL, left_map[0], left_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    imgR = cv2.remap(imgR, right_map[0], right_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    disparity = stereo.compute(imgL, imgR)
    print(disparity)

    cv2.imwrite("../Result/disparity_" + str(counter) + ".jpg", disparity)
    counter += 1
