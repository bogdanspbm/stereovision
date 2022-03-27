import cv2
import Objects.CameraBufferCleaner as cbf
from Utils.StereoUtils import splitMergedImage
from time import sleep


# Camera class based on VideoCapture + CameraBufferCleaner

class VirtualCamera():
    def __init__(self, name, width=0, heigth=0):
        self.name = name
        self.capture = cv2.VideoCapture(name)

        if width != 0 and heigth != 0:
            self.__setResolution(width, heigth)

        if self.capture.isOpened():
            print("Capture: " + str(name) + " is opened")
            self.buffer = cbf.CameraBufferCleanerThread(self.capture)
        else:
            self.buffer = None
            print("Capture: " + str(name) + " is failed")

    def getLastFrame(self):
        if self.buffer is not None:
            return self.buffer.last_frame

    def getSplittedFrames(self):
        if self.buffer is not None:
            return splitMergedImage(self.buffer.last_frame)

    def getLeftFrame(self):
        if self.buffer is not None:
            L, R = splitMergedImage(self.buffer.last_frame)
            return L

    def getRightFrame(self):
        if self.buffer is not None:
            L, R = splitMergedImage(self.buffer.last_frame)
            return R

    def __setResolution(self, width=1920, height=1080):
        if self.capture is not None and self.capture.isOpened():
            res = self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            print(res)
            res = self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print("Resolution: " + str(self.name) + " is updated")

    def saveLastFrame(self, path="frame.png", mode="BOTH"):
        last_frame = self.buffer.last_frame
        if last_frame is not None:
            if mode == "BOTH":
                cv2.imwrite(path, last_frame)
                print("Camera: " + str(self.name) + " - Last frame is saved")
            elif mode == "LEFT":
                cv2.imwrite(path, self.getLeftFrame())
                print("Camera: " + str(self.name) + " - Left frame is saved")
            elif mode == "RIGHT":
                cv2.imwrite(path, self.getRightFrame())
                print("Camera: " + str(self.name) + " - Right frame is saved")

    def showLastFrame(self, mode="BOTH"):

        # If buffer wasn't created return
        if self.buffer is None:
            return

        # Waiting for at least one frame
        i = 0
        last_frame = None

        while last_frame is None:
            last_frame = self.buffer.last_frame
            i += 1
            if last_frame is None:
                sleep(1)
            if i > 10:
                break

        # If frame exists show it
        if last_frame is not None:
            if mode == "BOTH":
                cv2.imshow("Virutal Camera" + str(self.name), last_frame)
            elif mode == "LEFT":
                cv2.imshow("Virutal Camera" + str(self.name), self.getLeftFrame())
            elif mode == "RIGHT":
                cv2.imshow("Virutal Camera" + str(self.name), self.getRightFrame())
            while True:
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
