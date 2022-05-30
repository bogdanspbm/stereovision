import time

'''
This is an implementation of labour timer
'''


class Timer():
    def __init__(self):
        self.last_time = time.time()

    '''
    Use this method to refresh timer to zero
    '''

    def refreshTimer(self):
        self.last_time = time.time()

    '''
    Use this method to get a time gone since the last call
    '''

    def getTime(self):
        res = time.time() - self.last_time
        self.last_time = time.time()
        return res

    '''
    Use this method to print a time gone since the last call
    '''

    def printTime(self, additional_string=""):
        print(additional_string, self.getTime())
