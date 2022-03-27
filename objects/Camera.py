import cv2
import objects.CameraBufferCleaner as cbf
import time


# Camera class based on VideoCapture + CameraBufferCleaner

class VirtualCamera():
    def __init__(self, name, width=0, heigth=0):
        self.name = name
        self.capture = cv2.VideoCapture(name)

        if width !=0 and heigth != 0:
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

    def __setResolution(self, width=1920, height=1080):
        if self.capture is not None and self.capture.isOpened():
            res = self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            print(res)
            res = self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print("Resolution: " + str(self.name) + " is updated")

    def showLastFrame(self):

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
                time.sleep(1)
            if i > 10:
                break

        # If frame exists show it
        if last_frame is not None:
            cv2.imshow("Virutal Camera" + str(self.name), last_frame)
            while True:
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
