import time

import cv2

import Objects.Camera as Camera

camera = Camera.VirtualCamera(0, 5148, 1088)

time.sleep(2)
camera.saveLastFrame(path="../Frames/frame.png", mode="LEFT")
# camera.showLastFrame("RIGHT")
