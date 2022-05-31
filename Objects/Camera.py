import cv2
import Objects.CameraBufferCleaner as cbf
from Utils.ImageUtils import splitMergedImage
from time import sleep

'''
This class implements a Virtual Camera
The constructor receives next parameters:
---------------------------------------------
name - a name of virtual camera
width - a desired resolution width 
height - a desired resolution height
'''


class VirtualCamera():

    def __init__(self, name, width=0, height=0):
        self.name = name
        self.capture = cv2.VideoCapture(name, cv2.CAP_MSMF)

        if width != 0 and height != 0:
            self.__setResolution(width, height)

        if self.capture.isOpened():
            print("Capture: " + str(name) + " is opened")
            self.buffer = cbf.CameraBufferCleanerThread(self.capture)
        else:
            self.buffer = None
            print("Capture: " + str(name) + " is failed")

    '''
    This method returns the last frame of this camera
    '''

    def getLastFrame(self):
        if self.buffer is not None:
            return self.buffer.last_frame

    '''
    This method returns a current frame of a camera buffer
    '''

    def getFrame(self):
        if self.capture is not None:
            ret, frame = self.capture.read()
            return frame

    '''
    This method returns the last frame split by left and right image
    '''

    def getSplittedFrames(self):
        if self.buffer is not None:
            return splitMergedImage(self.buffer.last_frame)

    '''
    This method returns only a left part of image
    '''

    def getLeftFrame(self):
        if self.buffer is not None:
            L, R = splitMergedImage(self.buffer.last_frame)
            return L

    '''
    This method returns only a right part of image
    '''

    def getRightFrame(self):
        if self.buffer is not None:
            L, R = splitMergedImage(self.buffer.last_frame)
            return R

    '''
    This method sets a desired image resolution
    '''

    def __setResolution(self, width=1920, height=1080):
        if self.capture is not None and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            print("Resolution: " + str(self.name) + " is updated")

    '''
    This method saves the last frame into Results folder with name frame.png
    '''

    def saveLastFrame(self, path="../Result/frame.png", mode="BOTH"):
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

    '''
    This method shows the last frame in an OpenCV window
    '''

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

    '''
    This method creates an OpenCV video stream for this virtual camera
    '''

    def showVideo(self, resize=(1920, 1080), filter="NONE", filter_params=[]):
        counter = 1
        tick = 1
        while True:
            tick += 1
            last_frame = self.getLastFrame()
            if last_frame is not None:
                if filter == "NONE":
                    resized = cv2.resize(last_frame, resize, interpolation=cv2.INTER_AREA)
                    cv2.imshow("Virutal Camera" + str(self.name), resized)
                elif filter == "CORNERS":
                    l_f, r_f = self.getSplittedFrames()
                    tmp_l = cv2.cvtColor(l_f, cv2.COLOR_BGR2GRAY)
                    tmp_r = cv2.cvtColor(r_f, cv2.COLOR_BGR2GRAY)
                    ret_l, corners_l = cv2.findChessboardCornersSB(tmp_l, filter_params[0])
                    ret_r, corners_r = cv2.findChessboardCornersSB(tmp_r, filter_params[0])

                    if (ret_l and ret_r) is True:
                        cv2.drawChessboardCorners(l_f, filter_params[0], corners_l, True)
                        cv2.drawChessboardCorners(r_f, filter_params[0], corners_r, True)
                        frame = cv2.hconcat([l_f, r_f])
                        resized = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                        cv2.imshow("Virutal Camera" + str(self.name), resized)
                    else:
                        frame = cv2.hconcat([tmp_l, tmp_r])
                        resized = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                        cv2.imshow("Virutal Camera" + str(self.name), resized)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                self.saveLastFrame(path="../Frames/frame_" + str(counter) + ".png")
                counter += 1
