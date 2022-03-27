import time

import cv2

import Objects.Camera as Camera

camera = Camera.VirtualCamera(0, 3840,1080)

time.sleep(2)
camera.saveLastFrame(path="../Frames/frame.png", mode="BOTH")
# camera.showLastFrame("RIGHT")
