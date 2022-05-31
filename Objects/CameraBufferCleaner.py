import threading

'''
This class implements an OpenCV camera buffer cleaner
It creates a thread which continuously reads the last frame of camera buffer
It allows to always save only the last frame 
---------------------------------------------
camera - an input camera buffer for cleaning
name - a name of created thread
'''


class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self). \
            __init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()
