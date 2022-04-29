from Utils.FileUtils import loadDistortionMatrix, loadCameraMatrix
from Utils.ImageUtils import splitMergedImage
import cv2
from Objects.Timer import Timer


class Undistorter():
    def __init__(self):
        self.undistort_left, self.undistort_right = loadDistortionMatrix()
        self.camera_left, self.camera_right = loadCameraMatrix()

    def quickUndistort(self, image):
        timer = Timer()
        image_left, image_right = splitMergedImage(image)
        und_left = cv2.undistort(image_left, self.camera_left, self.undistort_left)
        und_right = cv2.undistort(image_right, self.camera_right, self.undistort_right)
        timer.printTime("Undistort")
        return und_left, und_right


