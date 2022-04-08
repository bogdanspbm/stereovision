import Utils.StereoUtils as StereoUtils
import Utils.FileUtils as FileUtils
import cv2


def loadImagesAndGetDistance():
    P_1, P_2 = FileUtils.loadProjectionMatrix()
    images = FileUtils.getFramesImages()
    for image in images:
        image_left, image_right = StereoUtils.splitMergedImage(image)
        ret, corners_left = cv2.findChessboardCornersSB(image_left, (9, 14))
        ret, corners_right = cv2.findChessboardCornersSB(image_right, (9, 14))
        point_left = corners_left[(int)(len(corners_left) / 2)][0]
        point_right = corners_right[(int)(len(corners_right) / 2)][0]
        coord = StereoUtils.getWorldCoordinates(P_1, P_2, point_left, point_right)
        print(coord)


loadImagesAndGetDistance()
