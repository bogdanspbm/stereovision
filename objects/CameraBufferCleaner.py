import threading


# This class clears camera's capture buffer
# Default OpenCV Buffer can consist old images from camera
# To get the freshest image it's supposed to be cleaned

class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).\
            __init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()