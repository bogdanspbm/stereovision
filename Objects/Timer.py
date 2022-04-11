import time


class Timer():
    def __init__(self):
        self.last_time = time.time()

    def refreshTimer(self):
        self.last_time = time.time()

    def getTime(self):
        res = time.time() - self.last_time
        self.last_time = time.time()
        return res

    def printTime(self, additional_string = ""):
        print(additional_string, self.getTime())
